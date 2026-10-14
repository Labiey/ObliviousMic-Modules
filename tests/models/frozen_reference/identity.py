# Selected frozen offline reference bodies; supplied source license: plugins/models/LICENSE.txt.
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

EPSILON = 1e-12

SPEAKER_ENCODER_SAMPLE_RATE = 16000


class IdentityStreamError(RuntimeError):
    """The identity-stream pipeline violated its contract."""


@dataclass(frozen=True)
class IdentityStreamConfig:
    """Immutable configuration for the sliding-window identity stream.

    All fields are validated in ``__post_init__``.
    """

    sample_rate: int = 16000
    frame_samples: int = 160
    window_ms: int = 400
    hop_ms: int = 100
    silence_floor_dbfs: float = -55.0
    num_threads: int = 1

    def __post_init__(self) -> None:
        if self.sample_rate != 16000:
            raise IdentityStreamError("sample_rate must be 16000")
        if self.frame_samples != 160:
            raise IdentityStreamError("frame_samples must be 160")
        if not (200 <= self.window_ms <= 3000):
            raise IdentityStreamError("window_ms must be in [200, 3000]")
        if not (10 <= self.hop_ms <= self.window_ms):
            raise IdentityStreamError("hop_ms must be in [10, window_ms]")
        if self.num_threads <= 0:
            raise IdentityStreamError("num_threads must be positive")


def frame_energy_db(audio: np.ndarray, frame_samples: int = 160) -> np.ndarray:
    """10-ms-frame energy in dB: ``20 * log10(rms + eps)`` for each frame.

    Identical to ``presence_bsrnn_reference.frame_energy_db``.
    """
    if audio.ndim != 1:
        raise ValueError("audio must be mono 1-D")
    num_frames = max(1, audio.shape[0] // frame_samples)
    trimmed = audio[: num_frames * frame_samples]
    frames = trimmed.reshape(num_frames, frame_samples)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1) + EPSILON)
    return np.asarray(20.0 * np.log10(rms + EPSILON), dtype=np.float64)


def _window_rms_db(audio: np.ndarray) -> float:
    """Return RMS energy in dBFS for a 1-D audio segment."""
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if rms <= EPSILON:
        return -200.0
    return float(20.0 * math.log10(rms))


def sliding_identity(
    audio: np.ndarray,
    enrollment_embedding: np.ndarray,
    config: IdentityStreamConfig,
    embed: Callable[[np.ndarray], np.ndarray | None] | None = None,
) -> np.ndarray:
    """Compute strictly causal per-frame identity scores via sliding window.

    For each 10-ms frame *t*, the score is the cosine similarity between
    the enrollment embedding and the embedding of the most recent complete
    window whose end ≤ the end of frame *t*.  Windows whose RMS is below
    ``silence_floor_dbfs`` are skipped (the previous score is reused).
    The initial score is 0.0.

    Parameters
    ----------
    audio : np.ndarray
        1-D float32 mono audio at 16 kHz.
    enrollment_embedding : np.ndarray
        L2-normalized enrollment vector (float32).
    config : IdentityStreamConfig
        Window and silence parameters.
    embed : callable or None
        Function ``(audio_window) -> embedding | None``.  When ``None``,
        non-silent windows receive score 0.0 (useful for testing the
        sliding-window mechanics without an encoder).

    Returns
    -------
    np.ndarray
        Float64 array of length ``len(audio) // config.frame_samples``.
    """
    audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
    enrollment = np.ascontiguousarray(
        np.asarray(enrollment_embedding, dtype=np.float32).reshape(-1)
    )
    enr_norm = float(np.linalg.norm(enrollment))
    if enr_norm <= np.finfo(np.float32).eps:
        raise IdentityStreamError("enrollment embedding has zero norm")

    frame_samples = config.frame_samples
    window_samples = int(round(config.window_ms * config.sample_rate / 1000.0))
    hop_samples = int(round(config.hop_ms * config.sample_rate / 1000.0))
    n_frames = max(1, audio.shape[0] // frame_samples)

    scores = np.zeros(n_frames, dtype=np.float64)
    prev_score = 0.0

    # Each window k (0-indexed) spans [k*hop_samples, k*hop_samples+window_samples)
    # and ends at k*hop_samples + window_samples.
    # For each frame t, find the largest k with end <= (t+1)*frame_samples.

    window_idx = 0
    for t in range(n_frames):
        frame_end = (t + 1) * frame_samples

        # Advance window_idx to the last complete window ending ≤ frame_end
        best_k = -1
        while True:
            w_end = window_idx * hop_samples + window_samples
            if w_end > frame_end:
                break
            if w_end > audio.shape[0]:
                break
            best_k = window_idx
            window_idx += 1

        if best_k < 0:
            scores[t] = prev_score
            continue

        w_start = best_k * hop_samples
        w_end = w_start + window_samples
        window_audio = audio[w_start:w_end]

        if _window_rms_db(window_audio) < config.silence_floor_dbfs:
            scores[t] = prev_score
            continue

        if embed is None:
            scores[t] = prev_score
            continue

        vec = embed(window_audio)
        if vec is None:
            scores[t] = prev_score
            continue

        vec_norm = float(np.linalg.norm(vec))
        if vec_norm <= np.finfo(np.float32).eps:
            scores[t] = prev_score
            continue

        cosine = float(np.dot(vec, enrollment) / (vec_norm * enr_norm))
        scores[t] = cosine
        prev_score = cosine

    return scores
