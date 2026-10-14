# SPDX-License-Identifier: Apache-2.0
# Adapted for module-local distribution in 2026; see NOTICE and provenance.md.
# Reused from frozen OwnerMic; imports adapted only. Historical measurements in the
# original docstring describe the frozen version, not this refactor.
"""Torch-free streaming frame glue for the separation core (plan section 6.4).

The shipped runtime deliberately carries no torch, because the export chain that turns an
enrollment into a core stays on the development side.  That left the realtime path unable
to run a real core at all, since the existing wrapper imports torch at module scope.  This
module is the same contract implemented in numpy, frame for frame.

The structure is mirrored from :class:`ownermic.inference.streaming_bsrnn.StreamingBsrnn`
rather than re-derived, because the details are the specification:

* frames are centred windows of ``win`` samples stepping by ``hop``, addressed by absolute
  position, with whole-sample reflection at both ends of the stream;
* the analysis window is retained from ``frames_processed * hop - half - 1`` onward - that
  final ``- 1`` is the single sample the end-of-stream reflection reaches back for, which
  the torch wrapper used to get wrong and crash on exactly the common case;
* synthesis overlaps ``irfft(spectra) * window`` against an envelope of ``window ** 2``
  and the emitted region is divided by that envelope, so normalisation is exact rather
  than assumed;
* the backend performs the windowing and the STFT, so an exported core that takes
  ``mix_real`` / ``mix_imag`` slots in without changing the geometry.

Anything that would silently fabricate samples is an error here: a frame reaching outside
the retained history, a frame behind the emitted frontier, an envelope that vanishes
inside the emitted region, or a backend returning the wrong shape or non-finite values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: A backend turns ``[n, win]`` float32 frames plus carried states into complex spectra.
Layers = tuple[object, ...]


class FrameGlueError(RuntimeError):
    """Raised when the stream or a backend result cannot be honoured."""


@dataclass(frozen=True)
class FrameGeometry:
    """Analysis geometry in samples: an even, centred window and a hop no larger than it."""

    win: int
    hop: int

    def __post_init__(self) -> None:
        for name, value in (("win", self.win), ("hop", self.hop)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise FrameGlueError(f"{name} must be an integer number of samples")
            if value <= 0:
                raise FrameGlueError(f"{name} must be positive")
        if self.win % 2 != 0:
            raise FrameGlueError("win must be even for the centered STFT")
        if self.hop > self.win:
            raise FrameGlueError("hop must not exceed win")

    @property
    def half(self) -> int:
        return self.win // 2


def periodic_hann(win: int) -> np.ndarray:
    """The periodic Hann window, matching ``torch.hann_window(win)`` defaults.

    ``numpy.hanning`` is the *symmetric* variant (denominator ``win - 1``); mixing the two
    silently changes the overlap-add envelope, so the formula is written out here and the
    parity test pins it against torch.
    """
    if isinstance(win, bool) or not isinstance(win, int) or win <= 0:
        raise FrameGlueError("win must be a positive integer")
    index = np.arange(win, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * index / float(win))).astype(np.float32)


class FrameBackend:
    """Protocol implemented by the ONNX core adapter and by the parity fakes."""

    def initial_states(self) -> Layers:
        raise NotImplementedError

    def separate_frames(self, frames: np.ndarray, layers: Layers) -> tuple[np.ndarray, Layers]:
        """Return complex spectra of shape ``[frames, win // 2 + 1]`` and new states."""
        raise NotImplementedError


class NumpyFrameStream:
    """Chunked waveform in, chunked waveform out, using only numpy."""

    __slots__ = (
        "_backend",
        "_emitted",
        "_envelope",
        "_flushed",
        "_frames_processed",
        "_geometry",
        "_layers",
        "_ola_start",
        "_processed",
        "_signal",
        "_analysis",
        "_analysis_start",
        "_window",
        "_window_squared",
    )

    def __init__(self, *, backend: FrameBackend, win: int, hop: int) -> None:
        self._geometry = FrameGeometry(win=win, hop=hop)
        self._backend = backend
        self._window = periodic_hann(win)
        self._window_squared = self._window * self._window
        self._layers = self._initial_layers()
        self._analysis = np.zeros(0, dtype=np.float32)
        self._analysis_start = 0
        self._signal = np.zeros(0, dtype=np.float32)
        self._envelope = np.zeros(0, dtype=np.float32)
        self._ola_start = 0
        self._processed = 0
        self._emitted = 0
        self._frames_processed = 0
        self._flushed = False

    # -- public surface ----------------------------------------------------
    @property
    def geometry(self) -> FrameGeometry:
        return self._geometry

    @property
    def latency_samples(self) -> int:
        """Dependency horizon: an emitted sample needs input up to this far ahead."""
        return self._geometry.win

    @property
    def processed_samples(self) -> int:
        return int(self._processed)

    @property
    def emitted_samples(self) -> int:
        return int(self._emitted)

    @property
    def frames_processed(self) -> int:
        return int(self._frames_processed)

    def reset(self) -> None:
        """Drop all stream and backend state; used by the section 6.1 switch protocol."""
        self._layers = self._initial_layers()
        self._analysis = np.zeros(0, dtype=np.float32)
        self._analysis_start = 0
        self._signal = np.zeros(0, dtype=np.float32)
        self._envelope = np.zeros(0, dtype=np.float32)
        self._ola_start = 0
        self._processed = 0
        self._emitted = 0
        self._frames_processed = 0
        self._flushed = False

    def process(self, chunk: np.ndarray) -> np.ndarray:
        samples = self._as_samples(chunk)
        if self._flushed:
            raise FrameGlueError("stream already flushed; call reset() before processing")
        if samples.size == 0:
            return np.empty(0, dtype=np.float32)
        self._append(samples)
        return self._advance(flushed=False)

    def flush(self) -> np.ndarray:
        """Finish the stream, applying the reference end reflection."""
        if self._flushed:
            raise FrameGlueError("stream already flushed")
        self._flushed = True
        return self._advance(flushed=True)

    # -- stream bookkeeping ------------------------------------------------
    def _initial_layers(self) -> Layers:
        layers = self._backend.initial_states()
        if not isinstance(layers, tuple):
            raise FrameGlueError("backend.initial_states() must return a tuple")
        return layers

    @staticmethod
    def _as_samples(chunk: np.ndarray) -> np.ndarray:
        samples = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if samples.size and not np.all(np.isfinite(samples)):
            raise FrameGlueError("frame glue received non-finite samples")
        return samples

    def _append(self, samples: np.ndarray) -> None:
        if self._analysis.size == 0:
            self._analysis = samples.copy()
            self._analysis_start = self._processed
        else:
            expected = self._analysis_start + int(self._analysis.size)
            if expected != self._processed:
                raise FrameGlueError(
                    f"analysis history ends at {expected} but the stream is at {self._processed}; "
                    "samples were skipped or replayed without a reset"
                )
            self._analysis = np.concatenate((self._analysis, samples))
        self._processed += int(samples.size)

    def _advance(self, *, flushed: bool) -> np.ndarray:
        geometry = self._geometry
        half = geometry.half
        position = self._processed
        last_due = -1
        if position >= half + 1:
            last_due = (position - half) // geometry.hop
        if flushed:
            last_due = (1 + position // geometry.hop) - 1
        if last_due >= self._frames_processed:
            frames = self._build_frames(self._frames_processed, last_due + 1, position)
            spectra, layers = self._backend.separate_frames(frames, self._layers)
            expected = (last_due + 1 - self._frames_processed, half + 1)
            array = np.asarray(spectra)
            if not np.iscomplexobj(array):
                raise FrameGlueError("backend must return complex spectra")
            if array.shape != expected:
                raise FrameGlueError(f"backend returned spectra {array.shape}, expected {expected}")
            if array.size and not np.all(np.isfinite(array)):
                raise FrameGlueError("backend returned non-finite spectra")
            if not isinstance(layers, tuple) or len(layers) != len(self._layers):
                raise FrameGlueError("backend must return the same number of state layers")
            self._layers = layers
            self._add_frames(array.astype(np.complex64, copy=False), self._frames_processed)
            self._frames_processed = last_due + 1

        if flushed:
            limit = half + geometry.hop * (position // geometry.hop)
        else:
            limit = self._frames_processed * geometry.hop
        values = self._emit(limit)
        self._trim_analysis(position, flushed=flushed)
        return values

    def _build_frames(self, start_frame: int, end_frame: int, total: int) -> np.ndarray:
        geometry = self._geometry
        half = geometry.half
        if total < half + 2:
            raise FrameGlueError(
                "stream is shorter than one analysis window; the reference STFT's reflection "
                "padding is undefined for this length"
            )
        padded = (
            np.arange(start_frame, end_frame, dtype=np.int64)[:, None] * geometry.hop
            + np.arange(geometry.win, dtype=np.int64)[None, :]
        )
        source = np.where(
            padded < half,
            half - padded,
            np.where(padded < total + half, padded - half, 2 * total + half - 2 - padded),
        )
        if bool((source < self._analysis_start).any()) or bool((source >= total).any()):
            raise FrameGlueError(
                "frame support reaches outside the analysis buffer; streaming state is corrupt"
            )
        frames = self._analysis[source - self._analysis_start]
        return np.ascontiguousarray(frames, dtype=np.float32)

    def _add_frames(self, spectra: np.ndarray, start_frame: int) -> None:
        geometry = self._geometry
        win = geometry.win
        segments = (np.fft.irfft(spectra, n=win, axis=-1) * self._window).astype(np.float32)
        origin = self._ola_start
        for offset in range(int(spectra.shape[0])):
            position = (start_frame + offset) * geometry.hop
            if position < origin:
                raise FrameGlueError("overlap-add frame falls behind the emitted frontier")
            required_end = position + win
            span = required_end - origin
            if span > self._signal.size:
                missing = span - self._signal.size
                self._signal = np.concatenate((self._signal, np.zeros(missing, dtype=np.float32)))
                self._envelope = np.concatenate(
                    (self._envelope, np.zeros(missing, dtype=np.float32))
                )
            low = position - origin
            self._signal[low : low + win] += segments[offset]
            self._envelope[low : low + win] += self._window_squared

    def _emit(self, limit: int) -> np.ndarray:
        geometry = self._geometry
        first = geometry.half + self._emitted
        if limit <= first:
            return np.empty(0, dtype=np.float32)
        low = first - self._ola_start
        high = limit - self._ola_start
        if low < 0 or high > self._signal.size:
            raise FrameGlueError("emission window falls outside the overlap-add buffer")
        envelope = self._envelope[low:high]
        if bool((envelope <= 0).any()):
            raise FrameGlueError("overlap-add envelope vanished inside the emitted region")
        values = (self._signal[low:high] / envelope).astype(np.float32)
        self._signal = self._signal[high:].copy()
        self._envelope = self._envelope[high:].copy()
        self._ola_start = limit
        self._emitted += int(high - low)
        return values

    def _trim_analysis(self, position: int, *, flushed: bool) -> None:
        if flushed:
            self._analysis = np.zeros(0, dtype=np.float32)
            self._analysis_start = position
            return
        if self._frames_processed == 0:
            return
        # The "- 1" keeps the single sample the final reflected frame reaches back for.
        keep_from = max(0, self._frames_processed * self._geometry.hop - self._geometry.half - 1)
        drop = min(max(0, keep_from - self._analysis_start), int(self._analysis.size))
        if drop:
            self._analysis = self._analysis[drop:]
            self._analysis_start += drop


__all__ = [
    "FrameBackend",
    "FrameGeometry",
    "FrameGlueError",
    "NumpyFrameStream",
    "periodic_hann",
]
