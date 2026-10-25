# Reused from frozen OwnerMic; imports adapted only. Historical measurements in the
# original docstring describe the frozen version, not this refactor.
"""Streaming, frame-synchronous presence gate for the realtime audio engine.

Offline reference: ``evaluation.presence_gate_sim`` (``GateTuningConfig`` +
``simulate_gate``).  Realtime twin: this module (``PresenceGateConfig`` +
``StreamingPresenceGate``).  The two implementations MUST make the same
open/close decision on every frame for the same feature stream and may never
be modified one-sidedly: ``tests/test_target_presence.py`` pins them together
frame-by-frame (deterministic scenarios, a config grid, seeded random
sequences, prefix invariance and reset isolation).

The backend package never imports ``evaluation`` — that package is not
installable and is only reachable through PYTHONPATH, so importing it from
backend code would invert the dependency layers.  The config contract is
therefore redeclared here with identical field names, defaults and
validation, and the two copies are bound together exclusively by the parity
tests in ``tests/test_target_presence.py``.

This module imports only the standard library.  ``StreamingPresenceGate`` is
designed for the realtime processing worker thread (see
``ownermic.audio.engine.RealtimeMonoProcessor`` for the surrounding
interface style this class will be wrapped into): ``update`` is O(1) per
frame, performs no numpy vectorisation and allocates nothing, touches no
files, takes no locks and never sleeps.  State is kept exclusively in scalar
floats/ints/bools — the asymmetric EMA needs only the previous smoothed
value (no history window beyond that), and the attack/release/min-duration
logic needs only its counters.
"""

from __future__ import annotations

from dataclasses import dataclass


class PresenceGateError(RuntimeError):
    """A presence-gate configuration violated the state-machine contract."""


@dataclass(frozen=True, slots=True)
class PresenceGateConfig:
    """Immutable configuration for the presence-gate state machine.

    Field names, defaults and validation mirror
    ``evaluation.presence_gate_sim.GateTuningConfig`` exactly (locked by
    ``tests/test_target_presence.py``).  All thresholds are probabilities in
    [0, 1] unless noted otherwise.  ``ema_open_alpha``/``ema_close_alpha``
    control the asymmetric EMA speed; ``attack_frames``/``release_frames``
    are hysteresis counters; ``min_open_frames``/``min_close_frames``
    enforce minimum state durations; ``fast_open_score``/
    ``fast_open_energy_db`` provide a fast path for obvious owner speech;
    ``veto_score`` gates on an external identity check (e.g. CAM++).
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
    #: Sustained-silence close rule (switch-side, see presence_switch): audio
    #: below ``silence_close_energy_db`` for ``silence_close_ms`` cannot contain
    #: the owner, so those frames are decided absent instead of held.  The floor
    #: sits between game-audio leak into the mic (~-43 dBFS, where identity
    #: scores on the leak already close the gate) and room silence (~-55 dBFS
    #: and below, where the identity arm returns no observation at all).
    silence_close_energy_db: float = -48.0
    silence_close_ms: float = 500.0

    def __post_init__(self) -> None:
        if not (0.0 < self.close_threshold < self.open_threshold < 1.0):
            raise PresenceGateError(
                f"close_threshold ({self.close_threshold}) must be < "
                f"open_threshold ({self.open_threshold}) and both in (0, 1)"
            )
        if not (0.0 < self.ema_open_alpha <= 1.0):
            raise PresenceGateError(f"ema_open_alpha ({self.ema_open_alpha}) must be in (0, 1]")
        if not (0.0 < self.ema_close_alpha <= 1.0):
            raise PresenceGateError(f"ema_close_alpha ({self.ema_close_alpha}) must be in (0, 1]")
        if self.attack_frames < 0:
            raise PresenceGateError(f"attack_frames ({self.attack_frames}) must be non-negative")
        if self.release_frames < 0:
            raise PresenceGateError(f"release_frames ({self.release_frames}) must be non-negative")
        if self.min_open_frames < 0:
            raise PresenceGateError(
                f"min_open_frames ({self.min_open_frames}) must be non-negative"
            )
        if self.min_close_frames < 0:
            raise PresenceGateError(
                f"min_close_frames ({self.min_close_frames}) must be non-negative"
            )
        if not (self.fast_open_score <= self.open_threshold):
            raise PresenceGateError(
                f"fast_open_score ({self.fast_open_score}) must be <= "
                f"open_threshold ({self.open_threshold})"
            )
        if not (self.veto_score < self.close_threshold):
            raise PresenceGateError(
                f"veto_score ({self.veto_score}) must be < close_threshold ({self.close_threshold})"
            )
        if not -100.0 <= float(self.silence_close_energy_db) <= 0.0:
            raise PresenceGateError(
                f"silence_close_energy_db ({self.silence_close_energy_db}) "
                "must be in [-100, 0] dBFS"
            )
        if float(self.silence_close_ms) < 0.0:
            raise PresenceGateError(
                f"silence_close_ms ({self.silence_close_ms}) must be non-negative"
            )


class StreamingPresenceGate:
    """Frame-synchronous streaming twin of ``simulate_gate``.

    Consume one 10 ms frame per ``update`` call.  ``update`` returns the gate
    decision **for the current frame** (the same value ``simulate_gate``
    stores at ``gate[t]`` after consuming frame ``t``), so an opening frame
    returns True and a closing frame returns False.

    Semantics per frame, mirroring the offline reference exactly:

    1. Advance the asymmetric identity EMA (rising -> ``ema_open_alpha``,
       falling -> ``ema_close_alpha``; first sample seeds the filter).
    2. Advance the veto EMA the same way when a veto sample is provided.
    3. ``candidate_open`` <- (smoothed >= open_threshold) OR fast path, and
       not veto-active; ``candidate_close`` <- (smoothed <= close_threshold)
       OR veto-active.  Veto overrides the open candidates; when both
       candidates hold, open wins.
    4. Attack/release counters accumulate while a candidate holds and the
       opposite state's minimum duration is satisfied.
    5. ``min_open_frames``/``min_close_frames`` are enforced after a flip
       (flip-frame counters start at 1; a veto-forced close waives the
       ``min_close_frames`` hold but ``min_open_frames`` is always
       respected).

    ``identity_veto=None`` means "no veto observation this frame": the veto
    EMA is not advanced, and veto-active is evaluated against the most
    recent smoothed veto if the channel has ever been fed (a channel that
    never receives a sample is inactive forever).  ``mixture_energy_db`` is
    accepted for interface symmetry with the engine/feature layer but does
    not participate in the gate decision (as in the offline reference).

    All internal state is scalar and lives on the instance; ``reset()``
    returns the gate to its initial condition so the same stream can be
    replayed.  This class is not thread-safe by design: it belongs to one
    processing worker thread.
    """

    __slots__ = (
        "_config",
        "_frame_hz",
        "_open",
        "_consecutive_open",
        "_consecutive_closed",
        "_attack_count",
        "_release_count",
        "_closed_due_to_veto",
        "_identity_smoothed",
        "_veto_smoothed",
    )

    def __init__(self, config: PresenceGateConfig, *, frame_hz: int) -> None:
        if frame_hz <= 0:
            raise ValueError(f"frame_hz must be positive, got {frame_hz}")
        self._config = config
        self._frame_hz = int(frame_hz)
        self.reset()

    @property
    def config(self) -> PresenceGateConfig:
        """The immutable tuning configuration this gate was built with."""
        return self._config

    @property
    def frame_hz(self) -> int:
        """Analysis-frame rate in frames per second (100 -> 10 ms frames)."""
        return self._frame_hz

    @property
    def open(self) -> bool:
        """Current gate state after the most recent ``update`` (closed at start)."""
        return self._open

    @property
    def confidence(self) -> float | None:
        """Smoothed identity score after the most recent ``update``.

        ``None`` before the first ``update`` (or right after ``reset()``);
        afterwards the causally smoothed identity of the latest frame, which
        is the value the gate compares against ``open_threshold``/
        ``close_threshold``.  Intended for target-confidence monitoring.
        """
        return self._identity_smoothed

    def reset(self) -> None:
        """Return to the initial condition (gate closed, filters unseeded).

        Matches the offline reference's initialisation: the first frame may
        open immediately (the initial closed period is seeded with
        ``min_close_frames + 1`` consecutive closed frames).
        """
        self._open = False
        self._consecutive_open = 0
        self._consecutive_closed = self._config.min_close_frames + 1
        self._attack_count = 0
        self._release_count = 0
        self._closed_due_to_veto = False
        self._identity_smoothed = None
        self._veto_smoothed = None

    def update(
        self,
        identity_score: float,
        mixture_energy_db: float,
        output_energy_db: float,
        identity_veto: float | None = None,
    ) -> bool:
        """Consume one frame of features and return whether the gate is open.

        O(1) scalar work, no allocation, no I/O, no locking, no sleeping —
        safe for the realtime processing worker thread.
        """
        config = self._config

        # 1. Causal asymmetric identity EMA (same recurrence as the offline
        #    _asymmetric_ema, computed in plain Python floats so both sides
        #    round identically).
        x = float(identity_score)
        smoothed = self._identity_smoothed
        if smoothed is None:
            smoothed = x
        elif x > smoothed:
            smoothed = config.ema_open_alpha * x + (1.0 - config.ema_open_alpha) * smoothed
        else:
            smoothed = config.ema_close_alpha * x + (1.0 - config.ema_close_alpha) * smoothed
        self._identity_smoothed = smoothed

        # 2. Veto channel: advance its EMA only when a sample is present.
        if identity_veto is not None:
            v = float(identity_veto)
            veto_smoothed = self._veto_smoothed
            if veto_smoothed is None:
                veto_smoothed = v
            elif v > veto_smoothed:
                veto_smoothed = (
                    config.ema_open_alpha * v + (1.0 - config.ema_open_alpha) * veto_smoothed
                )
            else:
                veto_smoothed = (
                    config.ema_close_alpha * v + (1.0 - config.ema_close_alpha) * veto_smoothed
                )
            self._veto_smoothed = veto_smoothed
        veto_active = self._veto_smoothed is not None and self._veto_smoothed < config.veto_score

        # 3. Raw fast path + candidate open/close (open wins the conflict).
        output_db = float(output_energy_db)
        fast_open = x >= config.fast_open_score and output_db >= config.fast_open_energy_db
        candidate_open = (smoothed >= config.open_threshold or fast_open) and not veto_active
        candidate_close = smoothed <= config.close_threshold or veto_active
        if candidate_open and candidate_close:
            candidate_close = False

        # 4/5. State machine (strictly sequential, mirroring simulate_gate).
        if not self._open:
            self._consecutive_closed += 1
            if candidate_open:
                min_close_ok = (
                    self._consecutive_closed > config.min_close_frames or self._closed_due_to_veto
                )
                if min_close_ok:
                    self._attack_count += 1
                    if self._attack_count >= config.attack_frames:
                        self._open = True
                        self._consecutive_open = 1
                        self._attack_count = 0
                        self._closed_due_to_veto = False
                else:
                    self._attack_count = 0
            else:
                self._attack_count = 0
        else:
            self._consecutive_open += 1
            if candidate_close:
                if self._consecutive_open > config.min_open_frames:
                    self._release_count += 1
                    if self._release_count >= config.release_frames:
                        self._open = False
                        self._consecutive_closed = 1
                        self._release_count = 0
                        self._closed_due_to_veto = veto_active
                else:
                    self._release_count = 0
            else:
                self._release_count = 0

        return self._open
