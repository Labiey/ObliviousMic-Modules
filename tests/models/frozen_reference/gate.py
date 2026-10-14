# Selected frozen offline reference bodies; supplied source license: plugins/gate/LICENSE.txt.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FRAME_SAMPLES = 160
FRAME_DURATION_MS = 10.0


class GateSimulationError(RuntimeError):
    """The presence-gate simulation violated its contract."""


@dataclass(frozen=True)
class GateTuningConfig:
    """Immutable configuration for the presence-gate state machine.

    All thresholds are probabilities in [0,1] unless noted otherwise.
    ``ema_open_alpha`` / ``ema_close_alpha`` control the asymmetric EMA
    speed; ``attack_frames`` / ``release_frames`` are hysteresis counters.
    ``min_open_frames`` / ``min_close_frames`` enforce minimum state
    durations.  ``fast_open_score`` / ``fast_open_energy_db`` provide a
    fast-path for obvious owner speech.  ``veto_score`` is used to gate on
    an external identity check (e.g. CAM++).
    """

    open_threshold: float = 0.55
    close_threshold: float = 0.35
    ema_open_alpha: float = 0.5
    ema_close_alpha: float = 0.08
    attack_frames: int = 1
    release_frames: int = 20
    min_open_frames: int = 30
    min_close_frames: int = 20
    fast_open_score: float = 0.45
    fast_open_energy_db: float = -12.0
    veto_score: float = 0.15
    # Switch-side sustained-silence close rule (see presence_switch.py): the
    # offline gate state machine itself does not use these - they mirror the
    # backend fields so the config contract stays field-for-field equal.
    silence_close_energy_db: float = -48.0
    silence_close_ms: float = 500.0

    def __post_init__(self) -> None:
        if not (0.0 < self.close_threshold < self.open_threshold < 1.0):
            raise GateSimulationError(
                f"close_threshold ({self.close_threshold}) must be < "
                f"open_threshold ({self.open_threshold}) and both in (0, 1)"
            )
        if not (0.0 < self.ema_open_alpha <= 1.0):
            raise GateSimulationError(f"ema_open_alpha ({self.ema_open_alpha}) must be in (0, 1]")
        if not (0.0 < self.ema_close_alpha <= 1.0):
            raise GateSimulationError(f"ema_close_alpha ({self.ema_close_alpha}) must be in (0, 1]")
        if self.attack_frames < 0:
            raise GateSimulationError(f"attack_frames ({self.attack_frames}) must be non-negative")
        if self.release_frames < 0:
            raise GateSimulationError(
                f"release_frames ({self.release_frames}) must be non-negative"
            )
        if self.min_open_frames < 0:
            raise GateSimulationError(
                f"min_open_frames ({self.min_open_frames}) must be non-negative"
            )
        if self.min_close_frames < 0:
            raise GateSimulationError(
                f"min_close_frames ({self.min_close_frames}) must be non-negative"
            )
        if not (self.fast_open_score <= self.open_threshold):
            raise GateSimulationError(
                f"fast_open_score ({self.fast_open_score}) must be <= "
                f"open_threshold ({self.open_threshold})"
            )
        if not (self.veto_score < self.close_threshold):
            raise GateSimulationError(
                f"veto_score ({self.veto_score}) must be < close_threshold ({self.close_threshold})"
            )


@dataclass(frozen=True)
class GateFrameFeatures:
    """Per-frame features for one utterance.

    All array fields must have the same length (number of 10‑ms frames).
    ``output_samples`` is optional; when provided its length must equal
    ``len(identity_score) * FRAME_SAMPLES``.
    """

    identity_score: np.ndarray
    mixture_energy_db: np.ndarray
    output_energy_db: np.ndarray
    target_active: np.ndarray
    identity_veto: np.ndarray | None = None
    output_samples: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = len(self.identity_score)
        for name in ("mixture_energy_db", "output_energy_db", "target_active"):
            arr = getattr(self, name)
            if len(arr) != n:
                raise GateSimulationError(f"{name} length {len(arr)} != identity_score length {n}")
        if self.identity_veto is not None and len(self.identity_veto) != n:
            raise GateSimulationError(
                f"identity_veto length {len(self.identity_veto)} != identity_score length {n}"
            )
        if self.output_samples is not None:
            expected = n * FRAME_SAMPLES
            if len(self.output_samples) != expected:
                raise GateSimulationError(
                    f"output_samples length {len(self.output_samples)} != {expected} "
                    f"({n} frames × {FRAME_SAMPLES} samples)"
                )


def _asymmetric_ema(signal: np.ndarray, config: GateTuningConfig) -> np.ndarray:
    """Causal asymmetric EMA: faster when rising, slower when falling.

    ``s[0] = signal[0]``; for ``t > 0``:
      - if ``signal[t] > s[t-1]`` → ``alpha = ema_open_alpha``
      - else → ``alpha = ema_close_alpha``
      - ``s[t] = alpha * signal[t] + (1 - alpha) * s[t-1]``
    """
    n = len(signal)
    if n == 0:
        return np.array([], dtype=np.float64)
    smoothed = np.empty(n, dtype=np.float64)
    s = float(signal[0])
    smoothed[0] = s
    for t in range(1, n):
        x = float(signal[t])
        alpha = config.ema_open_alpha if x > s else config.ema_close_alpha
        s = alpha * x + (1.0 - alpha) * s
        smoothed[t] = s
    return smoothed


def smooth_identity(features: GateFrameFeatures, config: GateTuningConfig) -> np.ndarray:
    """Return the causally smoothed identity score (asymmetric EMA)."""
    return _asymmetric_ema(features.identity_score, config)


def _smooth_veto(features: GateFrameFeatures, config: GateTuningConfig) -> np.ndarray | None:
    """Smooth the veto signal if present, otherwise return None."""
    if features.identity_veto is None:
        return None
    return _asymmetric_ema(features.identity_veto, config)


def simulate_gate(features: GateFrameFeatures, config: GateTuningConfig) -> np.ndarray:
    """Run the causal state machine and return a boolean array (open=True).

    Semantics (strictly sequential per frame):
      1. Compute smoothed identity (causal EMA).
      2. Compute smoothed veto if provided.
      3. Determine candidate_open / candidate_close:
         - candidate_open ← (smoothed ≥ open_threshold) OR fast-path
         - candidate_close ← (smoothed ≤ close_threshold) OR veto-active
         - veto overrides candidate_open (even fast-path).
      4. Attack / release counters accumulate while candidate holds and
         the opposite state's minimum duration is satisfied.
      5. min_open_frames / min_close_frames are enforced after a flip.
      6. Veto can force a close without waiting for min_close to expire
         (but min_open is always respected).
    """
    n = len(features.identity_score)
    if n == 0:
        return np.array([], dtype=bool)

    smoothed = smooth_identity(features, config)
    veto_smoothed = _smooth_veto(features, config)

    gate = np.zeros(n, dtype=bool)
    state = False  # CLOSED
    consecutive_open = 0
    # Initial closed state must allow immediate opening; seed with a value
    # that already satisfies the min_close constraint.
    consecutive_closed = config.min_close_frames + 1
    attack_count = 0
    release_count = 0
    closed_due_to_veto = False

    for t in range(n):
        veto_active = False
        if veto_smoothed is not None and veto_smoothed[t] < config.veto_score:
            veto_active = True

        fast_open = (
            features.identity_score[t] >= config.fast_open_score
            and features.output_energy_db[t] >= config.fast_open_energy_db
        )
        candidate_open = (smoothed[t] >= config.open_threshold or fast_open) and not veto_active
        candidate_close = smoothed[t] <= config.close_threshold or veto_active

        # Resolve conflict: candidate_open wins over candidate_close
        if candidate_open and candidate_close:
            candidate_close = False

        if not state:
            # CLOSED
            consecutive_closed += 1
            if candidate_open:
                min_close_ok = consecutive_closed > config.min_close_frames or closed_due_to_veto
                if min_close_ok:
                    attack_count += 1
                    if attack_count >= config.attack_frames:
                        state = True
                        consecutive_open = 1
                        attack_count = 0
                        closed_due_to_veto = False
                else:
                    attack_count = 0
            else:
                attack_count = 0
        else:
            # OPEN
            consecutive_open += 1
            if candidate_close:
                if consecutive_open > config.min_open_frames:
                    release_count += 1
                    if release_count >= config.release_frames:
                        state = False
                        consecutive_closed = 1
                        release_count = 0
                        closed_due_to_veto = veto_active
                else:
                    release_count = 0
            else:
                release_count = 0

        gate[t] = state

    return gate
