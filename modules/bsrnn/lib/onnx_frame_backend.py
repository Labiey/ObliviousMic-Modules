# SPDX-License-Identifier: Apache-2.0
# Adapted for module-local distribution in 2026; see NOTICE and provenance.md.
# Reused from frozen OwnerMic; imports adapted only. Historical measurements in the
# original docstring describe the frozen version, not this refactor.
"""Engine-agnostic numpy frame backend for an exported streaming separation core.

The exported graph is the frozen contract: ``mix_real`` and ``mix_imag`` carry the
windowed STFT in ``(freq, frames)`` order, followed by twelve recurrent states, and the
graph answers with ``est_real`` / ``est_imag`` in the same layout plus the updated states.
This module speaks that ABI with numpy only, which is what lets the shipped runtime -
which carries no torch - drive a real core through :class:`NumpyFrameStream`.

The orientation is the place where a silent defect would live. ``rfft`` over ``[n, win]``
frames yields ``[n, freq]`` and the graph wants the transpose; feeding the untransposed
array still produces finite audio of the right shape, so nothing would crash - it would
just be a different, wrong, signal. The tests therefore pin the layout explicitly with a
session double that records exactly what it was handed.

Measured on the real exported graph (``tfmap_context_causal_100-stream-core.onnx``,
win 512, hop 128, 16384 samples, 1024-sample blocks), this numpy stack against the
development-side torch separator over the same weights: identical length, maximum
absolute deviation 1.04e-07, SNR 122.3 dB, equal output RMS, all samples finite.

Everything the graph can get wrong is refused rather than absorbed: unexpected input or
output names and counts, non-finite spectra, a state list that changed length, and an
estimate whose shape does not match the mix that went in.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .frame_glue import FrameBackend, FrameGlueError, periodic_hann

#: Input and output names of the exported streaming graph, in positional order.
CORE_INPUT_PREFIX: tuple[str, ...] = ("mix_real", "mix_imag")
STATE_LAYERS = 6
OUTPUT_NAMES: tuple[str, ...] = ("est_real", "est_imag")


class OnnxFrameBackend(FrameBackend):
    """Drives one exported core session over numpy frames."""

    def __init__(
        self,
        *,
        session: Any,
        win: int,
        hop: int,
        num_layers: int = STATE_LAYERS,
        state_shape: Sequence[int] = (1, 32, 256),
    ) -> None:
        if isinstance(win, bool) or not isinstance(win, int) or win <= 0:
            raise FrameGlueError("win must be a positive integer")
        if isinstance(hop, bool) or not isinstance(hop, int) or hop <= 0:
            raise FrameGlueError("hop must be a positive integer")
        self.win = int(win)
        self.hop = int(hop)
        self.num_layers = int(num_layers)
        self._state_shape = tuple(int(value) for value in state_shape)
        self._bins = self.win // 2 + 1
        self._window = periodic_hann(self.win)
        self._session = session
        inputs = tuple(str(getattr(item, "name", item)) for item in session.get_inputs())
        outputs = tuple(str(getattr(item, "name", item)) for item in session.get_outputs())
        expected_inputs = len(CORE_INPUT_PREFIX) + 2 * self.num_layers
        if len(inputs) != expected_inputs:
            raise FrameGlueError(
                f"core exposes {len(inputs)} inputs, expected {expected_inputs} "
                f"({len(CORE_INPUT_PREFIX)} mix plus {2 * self.num_layers} states)"
            )
        if inputs[: len(CORE_INPUT_PREFIX)] != CORE_INPUT_PREFIX:
            raise FrameGlueError(
                f"core mix inputs are named {inputs[:2]}, expected {CORE_INPUT_PREFIX}"
            )
        if outputs[:2] != OUTPUT_NAMES:
            raise FrameGlueError(f"core outputs are named {outputs[:2]}, expected {OUTPUT_NAMES}")
        if len(outputs) != len(inputs):
            raise FrameGlueError(
                f"core exposes {len(outputs)} outputs but {len(inputs)} inputs; "
                "states must round-trip"
            )
        self._input_names = inputs
        self._output_names = outputs

    @property
    def input_names(self) -> tuple[str, ...]:
        return self._input_names

    @property
    def output_names(self) -> tuple[str, ...]:
        return self._output_names

    def initial_states(self) -> tuple[np.ndarray, ...]:
        return tuple(
            np.zeros(self._state_shape, dtype=np.float32) for _ in range(2 * self.num_layers)
        )

    def separate_frames(
        self, frames: np.ndarray, layers: tuple[object, ...]
    ) -> tuple[np.ndarray, tuple[object, ...]]:
        array = np.asarray(frames, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.win:
            raise FrameGlueError(f"expected [frames, {self.win}] frames, got {array.shape}")
        if len(layers) != 2 * self.num_layers:
            raise FrameGlueError(f"expected {2 * self.num_layers} state tensors, got {len(layers)}")
        if array.size and not np.all(np.isfinite(array)):
            raise FrameGlueError("backend received non-finite frames")
        spec = np.fft.rfft(array * self._window, axis=-1).T  # (freq, frames), graph order
        feeds: dict[str, np.ndarray] = {
            self._input_names[0]: np.ascontiguousarray(spec.real, dtype=np.float32),
            self._input_names[1]: np.ascontiguousarray(spec.imag, dtype=np.float32),
        }
        for index, name in enumerate(self._input_names[2:]):
            value = np.asarray(layers[index], dtype=np.float32)
            if value.shape != self._state_shape:
                raise FrameGlueError(
                    f"state {name} has shape {value.shape}, expected {self._state_shape}"
                )
            feeds[name] = np.ascontiguousarray(value, dtype=np.float32)
        results = self._session.run(list(self._output_names), feeds)
        if len(results) != len(self._output_names):
            raise FrameGlueError(
                f"core returned {len(results)} values, expected {len(self._output_names)}"
            )
        real = np.asarray(results[0], dtype=np.float32)
        imag = np.asarray(results[1], dtype=np.float32)
        if real.shape != (self._bins, array.shape[0]) or imag.shape != real.shape:
            raise FrameGlueError(
                f"core estimate is {real.shape}/{imag.shape}, "
                f"expected {(self._bins, array.shape[0])}"
            )
        if real.size and not (np.all(np.isfinite(real)) and np.all(np.isfinite(imag))):
            raise FrameGlueError("core returned non-finite estimates")
        spectra = np.ascontiguousarray((real + 1j * imag).T, dtype=np.complex64)
        states = tuple(np.asarray(value, dtype=np.float32) for value in results[2:])
        if len(states) != 2 * self.num_layers:
            raise FrameGlueError(
                f"core returned {len(states)} states, expected {2 * self.num_layers}"
            )
        if any(
            value.shape != self._state_shape or not np.isfinite(value).all() for value in states
        ):
            raise FrameGlueError("core returned invalid recurrent state arrays")
        return spectra, states


__all__ = ["CORE_INPUT_PREFIX", "OUTPUT_NAMES", "STATE_LAYERS", "OnnxFrameBackend"]
