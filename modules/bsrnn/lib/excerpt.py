# SPDX-License-Identifier: Apache-2.0
# Adapted for module-local distribution in 2026; see NOTICE and provenance.md.
"""Select the enrollment excerpt that re-bakes the separation core.

The new enrollment flow: the owner records up to 60 s, the CAM++ embedding is
enrolled from it, then the *gate's own windowed scoring* runs back over the
recording to find where the owner is confidently present - continuous
presence above ``min_probability`` for at least ``min_duration_s`` - and the
3.0 s excerpt for the BSRNN re-bake is cut from the best such segment.  This
replaces the historical export's hardcoded 30-33 s window: the excerpt is now
chosen by identity evidence instead of by position.

Scoring uses the same ``WindowEmbeddingExtractor.embed`` chain as the realtime
identity arm, so "presence probability" here is exactly the cosine the gate
sees - 0.20 sits between the champion's close (0.12) and open (0.33)
thresholds, i.e. a stretch where the gate would lean open.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

#: The excerpt length the baked ``enroll_mag`` reference expects (3.0 s @ 16 kHz).
EXCERPT_SAMPLES: int = 48_000
SAMPLE_RATE: int = 16_000


class ExcerptSelectionError(RuntimeError):
    """Named failure: no qualifying owner-present segment in the recording."""


@dataclass(frozen=True)
class ExcerptSelection:
    start_sample: int
    samples: np.ndarray
    mean_score: float
    segment_start_sample: int
    segment_end_sample: int


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right))


def select_enrollment_excerpt(
    audio: np.ndarray,
    embedding: np.ndarray,
    *,
    embed: Callable[[np.ndarray], np.ndarray | None],
    window_ms: int = 600,
    hop_ms: int = 300,
    min_probability: float = 0.20,
    min_duration_s: float = 4.0,
) -> ExcerptSelection:
    """Find the best owner-present segment and cut the re-bake excerpt from it.

    ``embed`` maps an audio window to an L2-normalized embedding (``None`` on
    failure), injected so tests stay deterministic and model-free.  Raises
    :class:`ExcerptSelectionError` when no qualifying segment exists - the
    enrollment flow surfaces that as "record again", never a silent fallback
    to an arbitrary window.
    """
    audio = np.asarray(audio)
    # The recorder hands over [frames, channels] (mono is [N, 1]); the scoring
    # path is channel-agnostic mono, so a single-channel 2-D layout is the same
    # signal - squeeze it instead of refusing the recorder's own format.
    if audio.ndim == 2 and audio.shape[1] == 1:
        audio = audio[:, 0]
    if audio.ndim != 1 or audio.dtype != np.float32:
        raise ExcerptSelectionError("enrollment audio must be mono float32")
    target = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(target))
    if norm <= 0.0:
        raise ExcerptSelectionError("enrollment embedding has zero norm")

    window = int(window_ms * SAMPLE_RATE / 1000)
    hop = int(hop_ms * SAMPLE_RATE / 1000)
    positions = range(0, max(0, audio.shape[0] - window) + 1, hop)
    scores: list[tuple[int, float]] = []
    for start in positions:
        vector = embed(audio[start : start + window])
        if vector is None:
            continue
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.shape != target.shape:
            continue
        scores.append((start, _cosine(vector, target)))

    if not scores:
        raise ExcerptSelectionError(
            "no window of the enrollment audio could be scored; recording too short"
        )

    # Maximal runs of consecutive scored windows above the probability floor.
    runs: list[list[tuple[int, float]]] = []
    current: list[tuple[int, float]] = []
    for entry in scores:
        if entry[1] > min_probability:
            current.append(entry)
        else:
            if current:
                runs.append(current)
            current = []
    if current:
        runs.append(current)

    min_span = int(min_duration_s * SAMPLE_RATE)
    qualifying = [run for run in runs if run[-1][0] + window - run[0][0] >= min_span]
    if not qualifying:
        raise ExcerptSelectionError(
            "no owner-present segment of at least "
            f"{min_duration_s:.0f} s above {min_probability:.2f}; record again"
        )

    # Best segment: highest mean score, longest on ties.
    best = max(
        qualifying,
        key=lambda run: (
            sum(score for _, score in run) / len(run),
            run[-1][0] - run[0][0],
        ),
    )
    segment_start = best[0][0]
    segment_end = best[-1][0] + window

    # Cut the excerpt where the contained windows score highest on average.
    lookup = dict(best)
    audio_len = audio.shape[0]
    lo = max(0, min(segment_end - EXCERPT_SAMPLES, audio_len - EXCERPT_SAMPLES))
    hi = max(lo, min(segment_start, audio_len - EXCERPT_SAMPLES))
    best_cut, best_cut_score = None, -float("inf")
    for cut in range(lo, hi + 1, hop):
        inside = [
            score
            for start, score in best
            if start >= cut and start + window <= cut + EXCERPT_SAMPLES
        ]
        if not inside:
            continue
        mean = sum(inside) / len(inside)
        if mean > best_cut_score:
            best_cut, best_cut_score = cut, mean
    if best_cut is None:
        # Segment shorter than the excerpt in scored windows: centre the cut.
        best_cut = max(
            0,
            min(
                (segment_start + segment_end) // 2 - EXCERPT_SAMPLES // 2,
                audio_len - EXCERPT_SAMPLES,
            ),
        )
        best_cut_score = sum(score for _, score in best) / len(best)

    return ExcerptSelection(
        start_sample=best_cut,
        samples=np.ascontiguousarray(audio[best_cut : best_cut + EXCERPT_SAMPLES]),
        mean_score=best_cut_score,
        segment_start_sample=segment_start,
        segment_end_sample=segment_end,
    )
