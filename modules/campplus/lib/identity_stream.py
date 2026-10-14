# SPDX-License-Identifier: Apache-2.0
# Adapted for module-local distribution in 2026; see NOTICE and provenance.md.
# Reused from frozen OwnerMic; imports adapted only. Historical measurements in the
# original docstring describe the frozen version, not this refactor.
"""Causal per-frame identity-score arm for the presence switch (streaming).

This module is the realtime, backend-only twin of the offline identity arm in
``evaluation/presence_identity_stream.py``.  ``backend`` must never import
``evaluation`` (it is a test-only path), so the window semantics, energy
formulas and constants are reproduced here; ``tests/test_identity_stream.py``
pins the streaming provider frame-for-frame (bitwise, float64) against the
offline ``sliding_identity`` reference with a deterministic fake embedder.

Data path per 10 ms gate frame (``160`` samples at 16 kHz)::

    mixture frame -> SlidingIdentityProvider (this module)
        -> WindowEmbeddingExtractor.embed() (CPU ONNX CAM++ window encoder)
        -> IdentityObservation(score, veto, mixture_energy_db)
        -> ownermic.processing.presence_switch.PresenceIdentityProvider seam

Semantics reproduced from the offline reference (read from its source, not
paraphrased):

* window ``k`` covers ``[k*hop, k*hop + window)`` samples; at frame ``t`` the
  window used is the largest ``k`` with ``k*hop + window <= (t+1)*frame``;
* no complete window yet, window RMS below ``silence_floor_dbfs``,
  ``embed()`` returning ``None``, or a zero-norm embedding all **reuse the
  previous frame's score**; the initial score is ``0.0`` (never ``None``);
* the score is the raw, unstretched cosine similarity (it may be negative;
  this provider never clamps it into ``[0, 1]``);
* the enrollment embedding must be L2-normalized and a zero-norm enrollment
  is an error.

Realtime differences that are output-invisible for deterministic embedders:

* windows are embedded **once at the frame where they become complete**
  instead of being re-embedded on every plateau frame, and
* ``window_ms``/``hop_ms`` must be frame-aligned (multiples of ``10`` ms),
  which makes every newly complete window end exactly at the current frame
  end; a ring buffer of exactly ``window_samples`` then provably suffices and
  memory stays bounded (the offline reference may slice arbitrary look-back,
  which a streaming stage cannot do with bounded memory).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import numpy as np
import onnxruntime as ort

from .features import (
    SPEAKER_ENCODER_SAMPLE_RATE,
    compute_kaldi_fbank,
)
from .identity_seam import IdentityObservation

#: Same value as ``evaluation.presence_identity_stream.EPSILON``; keeping it
#: identical is what makes the silence floor mean the same thing online and
#: offline (the floor is applied to the same dBFS formula in both places).
EPSILON: Final[float] = 1e-12

#: Smallest practical windowing period in milliseconds (10 ms frame grid).
_GRID_MS: Final[int] = 10


class IdentityStreamError(RuntimeError):
    """The realtime identity-stream pipeline violated its contract."""


# ---------------------------------------------------------------------------
# Window embedding extractor (backend twin of the offline WindowSpeakerEncoder)
# ---------------------------------------------------------------------------


def _create_cpu_session(
    model_path: Path,
    num_threads: int,
    *,
    engine: str = "onnxruntime",
    device: str = "CPU",
) -> Any:
    """Build the identity session through the shared engine seam.

    The file and thread checks stay here because this arm reports failures as
    ``IdentityStreamError``, and the option set now lives in exactly one place,
    ``engine_sessions.ort_session_options``, which is the same set this function used to
    build by hand - so choosing an engine cannot silently move the operating point.
    """
    from .sessions import make_session

    if not Path(model_path).is_file():
        raise IdentityStreamError(f"speaker model does not exist: {model_path}")
    if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads <= 0:
        raise IdentityStreamError("num_threads must be a positive integer")
    return make_session(engine, model_path, num_threads, device=device)


class WindowEmbeddingExtractor:
    """Extract L2-normalized CAM++ embeddings from short audio windows.

    Backend twin of ``evaluation.presence_identity_stream.WindowSpeakerEncoder``
    with the same ``embed()`` chain (``compute_kaldi_fbank`` -> per-frame mean
    subtraction -> ONNX session -> L2 normalization, ``None`` on failure) but
    an **explicit** ``initialize()``: the session is created exactly once there
    and ``embed()`` never creates a session and never touches the disk.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        num_threads: int = 1,
        engine: str = "onnxruntime",
        device: str = "CPU",
        probe_window_samples: int | None = None,
        session_factory: Callable[[Path, int], ort.InferenceSession] | None = None,
    ) -> None:
        self._model_path = Path(model_path).resolve()
        self._num_threads = num_threads
        self._engine = engine
        self._device = device
        self._probe_window_samples = int(probe_window_samples or 0)
        self._session_factory: Callable[[Path, int], ort.InferenceSession] = session_factory or (
            lambda path, threads: _create_cpu_session(path, threads, engine=engine, device=device)
        )
        self._session: ort.InferenceSession | None = None
        self._input_name = ""
        self._output_name = ""

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def initialized(self) -> bool:
        return self._session is not None

    @property
    def num_threads(self) -> int:
        return self._num_threads

    @property
    def engine(self) -> str:
        """Engine this extractor was built for, so the API can report what is actually used."""
        return self._engine

    def _probe_engine(self) -> None:
        """Run one window of the length the pipeline will actually ask for.

        A failed identity inference is indistinguishable from a quiet room downstream: the
        provider reads "no evidence" and the gate stays closed forever without reporting
        anything. Measured on this machine, OpenVINO's CPU plugin cannot execute this CAM++
        graph at the shipped 600 ms window at all, while 3 s windows run and agree with
        onnxruntime to cosine 1.0 - exactly the shape of problem that has to be refused at
        construction, where a named error and a remedy still exist.
        """
        if self._probe_window_samples <= 0:
            return
        silent = np.zeros(self._probe_window_samples, dtype=np.float32)
        if self.embed(silent) is None:
            raise IdentityStreamError(
                f"the {self._engine} engine on device {self._device} cannot run the speaker "
                f"encoder on a {self._probe_window_samples}-sample window; pick engine "
                "onnxruntime or raise the identity window length"
            )

    def initialize(self) -> None:
        """Create the CPU session and parse its single input/output.

        Idempotent (mirrors ``OnnxSpeakerEncoder.initialize``); a missing model
        file is only an error when the default session factory is used, so
        tests may inject a fake factory and a path that does not exist.
        """
        if self._session is not None:
            return
        session = self._session_factory(self._model_path, self._num_threads)
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise IdentityStreamError("speaker model must have exactly one input and one output")
        self._session = session
        self._input_name = str(inputs[0].name)
        self._output_name = str(outputs[0].name)
        self._probe_engine()

    def embed(self, audio: np.ndarray) -> np.ndarray | None:
        """Embed one audio window (any duration >= one fbank frame).

        Returns ``None`` when the window is too short for one fbank frame, the
        session fails, or the embedding has zero norm.  Raises when called
        before :meth:`initialize`; never creates a session and never reads the
        model file.
        """
        if self._session is None:
            raise IdentityStreamError(
                "window embedding extractor is not initialized; call initialize() first"
            )
        audio_arr = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
        try:
            features = compute_kaldi_fbank(audio_arr, SPEAKER_ENCODER_SAMPLE_RATE)
        except Exception:
            return None
        if features.shape[0] == 0:
            return None
        features = features - features.mean(axis=0, keepdims=True)
        input_tensor = np.ascontiguousarray(features[np.newaxis, :, :], dtype=np.float32)
        try:
            outputs = self._session.run([self._output_name], {self._input_name: input_tensor})
        except Exception:
            return None
        vector = np.asarray(outputs[0], dtype=np.float32)
        if vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector[0]
        norm = float(np.linalg.norm(vector))
        if norm <= np.finfo(np.float32).eps:
            return None
        return np.ascontiguousarray(vector / norm, dtype=np.float32)

    def embed_enrollment(self, audio: np.ndarray) -> np.ndarray:
        """Enrollment convenience: embed and require at least 3 s of audio."""
        audio_arr = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
        min_samples = 3 * SPEAKER_ENCODER_SAMPLE_RATE
        if audio_arr.shape[0] < min_samples:
            raise IdentityStreamError(
                "enrollment audio must be at least 3 s; got "
                f"{audio_arr.shape[0] / SPEAKER_ENCODER_SAMPLE_RATE:.3f} s"
            )
        result = self.embed(audio_arr)
        if result is None:
            raise IdentityStreamError("failed to extract enrollment embedding")
        return result


# ---------------------------------------------------------------------------
# Offline-identical helpers
# ---------------------------------------------------------------------------


def _window_rms_db(audio: np.ndarray) -> float:
    """RMS energy in dBFS of a 1-D segment; identical to the offline helper."""
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if rms <= EPSILON:
        return -200.0
    return float(20.0 * math.log10(rms))


def _frame_energy_db_scalar(samples: np.ndarray) -> float:
    """One 10 ms frame's energy in dB; identical to the offline per-frame def.

    ``20 * log10(sqrt(mean(x^2) + EPSILON) + EPSILON)`` in float64, the same
    expression (and therefore bitwise-identical result) as
    ``presence_switch.frame_energy_db`` and the offline ``frame_energy_db``.
    """
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0:
        return float(20.0 * math.log10(EPSILON + EPSILON))
    rms = math.sqrt(float(np.mean(np.square(values))) + EPSILON)
    return float(20.0 * math.log10(rms + EPSILON))


def _validate_aligned_ms(name: str, value: object, low: int, high: int) -> None:
    """Validate *value* is an integer in ``[low, high]`` and frame-aligned."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise IdentityStreamError(f"{name} must be an integer in [{low}, {high}]; got {value!r}")
    if not low <= value <= high:
        raise IdentityStreamError(f"{name} must be in [{low}, {high}]; got {value!r}")
    if value % _GRID_MS:
        raise IdentityStreamError(
            f"{name} must be frame-aligned (a multiple of {_GRID_MS} ms) so streaming "
            f"windows end on frame boundaries; got {value!r}"
        )


class _SlidingWindowStream:
    """One sliding-window score stream (main or veto channel).

    Holds a bounded ring of exactly ``window_samples`` float32 samples in
    chronological order, the last assessed window index and the held score.
    """

    __slots__ = ("window_samples", "hop_samples", "ring", "filled", "last_k", "held")

    def __init__(self, window_samples: int, hop_samples: int) -> None:
        self.window_samples = window_samples
        self.hop_samples = hop_samples
        self.ring = np.zeros(window_samples, dtype=np.float32)
        self.filled = 0
        self.last_k = -1
        self.held = 0.0

    def store(self, frame: np.ndarray) -> None:
        """Append one 10 ms frame, keeping the ring at <= window_samples."""
        frame_len = frame.shape[0]
        capacity = self.window_samples
        if self.filled + frame_len <= capacity:
            self.ring[self.filled : self.filled + frame_len] = frame
            self.filled += frame_len
        else:
            self.ring[:-frame_len] = self.ring[frame_len:]
            self.ring[-frame_len:] = frame
            self.filled = capacity

    def reset(self) -> None:
        self.filled = 0
        self.last_k = -1
        self.held = 0.0


# ---------------------------------------------------------------------------
# Sliding identity provider
# ---------------------------------------------------------------------------


class SlidingIdentityProvider:
    """Realtime identity arm: one :class:`IdentityObservation` per gate frame.

    Reproduces ``evaluation.presence_identity_stream.sliding_identity``
    frame-for-frame while streaming, and satisfies the
    ``presence_switch.PresenceIdentityProvider`` seam (``observe``/``reset``/
    ``algorithmic_latency_ms``).  The embedder is injected and only needs an
    ``embed(audio) -> ndarray | None`` callable, so the tests can use a
    deterministic fake model; production wires a
    :class:`WindowEmbeddingExtractor`.

    Frame semantics (identical to the offline reference):

    * every ``observe()`` after :meth:`initialize` returns an observation -
      frames before the first complete window simply carry ``score == 0.0``.
      ``None`` is never used to mean "window not yet full" (the switch would
      hold its previous state and diverge from the offline run); ``observe()``
      before :meth:`initialize` is a programming error and raises.
    * ``score`` is the raw unstretched cosine (it may be negative) or the held
      previous score; the initial held score is ``0.0``.
    * ``mixture_energy_db`` is the energy of the current 160-sample frame,
      ``20*log10(sqrt(mean(x^2)+EPSILON)+EPSILON)`` in float64 - the offline
      per-frame energy formula.
    * ``veto``: when ``veto_window_ms`` is enabled it is the score of a second,
      independently sliding window of that length over the same frames (same
      hold/``0.0`` rules); when disabled (``None``) it is ``None`` per frame.

    Latency: :attr:`algorithmic_latency_ms` honestly reports the main window
    length plus one hop of alignment margin (see the property docstring).
    """

    def __init__(
        self,
        enrollment_embedding: np.ndarray,
        embedder: object | None = None,
        *,
        sample_rate: int = SPEAKER_ENCODER_SAMPLE_RATE,
        frame_samples: int = 160,
        window_ms: int = 600,
        hop_ms: int = 100,
        silence_floor_dbfs: float = -55.0,
        veto_window_ms: int | None = 600,
    ) -> None:
        if sample_rate != SPEAKER_ENCODER_SAMPLE_RATE:
            raise IdentityStreamError(f"sample_rate must be 16000; got {sample_rate!r}")
        if frame_samples != 160:
            raise IdentityStreamError(f"frame_samples must be 160; got {frame_samples!r}")
        _validate_aligned_ms("window_ms", window_ms, 200, 3000)
        _validate_aligned_ms("hop_ms", hop_ms, 10, window_ms)
        if veto_window_ms is not None:
            _validate_aligned_ms("veto_window_ms", veto_window_ms, 200, 3000)
            if veto_window_ms < hop_ms:
                raise IdentityStreamError(
                    f"veto_window_ms must be >= hop_ms; got {veto_window_ms!r} < {hop_ms!r}"
                )
        if isinstance(silence_floor_dbfs, bool) or not isinstance(silence_floor_dbfs, (int, float)):
            raise IdentityStreamError(
                f"silence_floor_dbfs must be a finite number; got {silence_floor_dbfs!r}"
            )
        if not math.isfinite(float(silence_floor_dbfs)):
            raise IdentityStreamError("silence_floor_dbfs must be finite")
        if embedder is not None and not callable(getattr(embedder, "embed", None)):
            raise IdentityStreamError("embedder must expose an embed(audio) callable")

        self._enrollment_raw = enrollment_embedding
        self._embedder: object | None = embedder
        self._sample_rate = int(sample_rate)
        self._frame_samples = int(frame_samples)
        self._window_ms = int(window_ms)
        self._hop_ms = int(hop_ms)
        self._silence_floor_dbfs = float(silence_floor_dbfs)
        self._veto_window_ms: int | None = None if veto_window_ms is None else int(veto_window_ms)

        self._window_samples = int(round(self._window_ms * self._sample_rate / 1000.0))
        self._hop_samples = int(round(self._hop_ms * self._sample_rate / 1000.0))
        self._veto_samples = (
            None
            if self._veto_window_ms is None
            else int(round(self._veto_window_ms * self._sample_rate / 1000.0))
        )

        self._initialized = False
        self._enrollment: np.ndarray | None = None
        self._enrollment_norm = 0.0
        self._main = _SlidingWindowStream(self._window_samples, self._hop_samples)
        self._veto = (
            None
            if self._veto_samples is None
            else _SlidingWindowStream(self._veto_samples, self._hop_samples)
        )
        self._frames = 0

    # ------------------------------------------------------------------
    # Configuration (read-only)
    # ------------------------------------------------------------------

    @property
    def window_ms(self) -> int:
        return self._window_ms

    @property
    def hop_ms(self) -> int:
        return self._hop_ms

    @property
    def veto_window_ms(self) -> int | None:
        return self._veto_window_ms

    @property
    def silence_floor_dbfs(self) -> float:
        return self._silence_floor_dbfs

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_samples(self) -> int:
        return self._frame_samples

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def embedder(self) -> object | None:
        return self._embedder

    @property
    def algorithmic_latency_ms(self) -> float:
        """Honest observation delay of this arm: window + one hop margin.

        A score reported at frame ``t`` may describe a window whose newest
        sample is the current frame's end, but window completion is quantized
        to the hop grid, so the oldest content of a just-completed window can
        lag the frame end by up to ``window_ms + hop_ms - frame``.  Adding the
        full hop as alignment margin keeps the claim safe for any frame within
        a hop (the switch separately adds its own one-frame alignment).
        """
        return float(self._window_ms + self._hop_ms)

    # ------------------------------------------------------------------
    # PresenceIdentityProvider protocol
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Validate the enrollment/embedder and initialize the embedder.

        The enrollment embedding must be L2-normalized with a nonzero norm
        (mirroring the offline zero-norm error); the embedder, when present,
        must expose ``embed(audio)``.  Raises when already initialized
        (matching ``presence_switch``/``streaming_processor`` conventions).
        """
        if self._initialized:
            raise IdentityStreamError("sliding identity provider is already initialized")
        if self._embedder is None:
            raise IdentityStreamError(
                "sliding identity provider requires an embedder exposing embed(audio)"
            )
        enrollment = np.ascontiguousarray(
            np.asarray(self._enrollment_raw, dtype=np.float32).reshape(-1)
        )
        enr_norm = float(np.linalg.norm(enrollment))
        if enr_norm <= np.finfo(np.float32).eps:
            raise IdentityStreamError("enrollment embedding has zero norm")
        embedder_initialize = getattr(self._embedder, "initialize", None)
        if callable(embedder_initialize):
            embedder_initialize()
        self._enrollment = enrollment
        self._enrollment_norm = enr_norm
        self._frames = 0
        self._initialized = True

    def observe(self, samples: np.ndarray, *, sample_rate: int) -> IdentityObservation:
        """Consume exactly one 10 ms gate frame and return its observation.

        Always returns an observation (never ``None``) once initialized;
        contract violations raise :class:`IdentityStreamError`.
        """
        if not self._initialized:
            raise IdentityStreamError(
                "sliding identity provider is not initialized; call initialize() first"
            )
        if sample_rate != self._sample_rate:
            raise IdentityStreamError(
                f"sliding identity provider requires {self._sample_rate} Hz audio; "
                f"got {sample_rate} Hz"
            )
        frame = np.ascontiguousarray(np.asarray(samples, dtype=np.float32).reshape(-1))
        if frame.shape[0] != self._frame_samples:
            raise IdentityStreamError(
                "sliding identity provider requires exactly one gate frame "
                f"({self._frame_samples} samples); got {frame.shape[0]}"
            )

        mixture_db = _frame_energy_db_scalar(frame)
        self._frames += 1
        frame_end = self._frames * self._frame_samples

        self._main.store(frame)
        score = self._advance_stream(self._main, frame_end)

        veto: float | None = None
        if self._veto is not None:
            self._veto.store(frame)
            veto_score = self._advance_stream(self._veto, frame_end)
            veto = float(veto_score)

        return IdentityObservation(
            score=float(score),
            veto=veto,
            mixture_energy_db=mixture_db,
        )

    def reset(self) -> None:
        """Clear both sliding streams; held scores return to the initial 0.0.

        Does not release initialization (mirrors the switch, whose reset keeps
        the processor initialized); the next ``observe()`` after a ``reset()``
        replays a fresh stream identical to a new run on the same audio.
        """
        if not self._initialized:
            raise IdentityStreamError(
                "sliding identity provider is not initialized; call initialize() first"
            )
        self._main.reset()
        if self._veto is not None:
            self._veto.reset()
        self._frames = 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _advance_stream(self, stream: _SlidingWindowStream, frame_end: int) -> float:
        """Score *stream* at the current frame end; identical to offline.

        ``stream.last_k`` is committed only after the window assessment
        completes, so an ``embed()`` exception (which propagates to the switch,
        that holds its state and records the error) leaves the window
        unassessed and it is retried on the next frame.
        """
        if frame_end < stream.window_samples:
            return stream.held
        best_k = (frame_end - stream.window_samples) // stream.hop_samples
        if best_k == stream.last_k:
            return stream.held
        embedder = self._embedder
        if embedder is None:  # pragma: no cover - initialize() enforces this
            return stream.held
        if stream.filled == stream.window_samples:
            # Frame-aligned windows complete exactly at a frame end, so the
            # window is always the trailing window_samples samples (the full
            # ring, in chronological order).
            if _window_rms_db(stream.ring) >= self._silence_floor_dbfs:
                vec = embedder.embed(stream.ring)
                if vec is not None:
                    vec_norm = float(np.linalg.norm(vec))
                    if vec_norm > np.finfo(np.float32).eps:
                        enrollment = self._enrollment
                        if enrollment is None:  # pragma: no cover - invariant
                            raise IdentityStreamError(
                                "sliding identity provider is not initialized"
                            )
                        cosine = float(np.dot(vec, enrollment) / (vec_norm * self._enrollment_norm))
                        stream.held = cosine
        stream.last_k = best_k
        return stream.held
