# Frozen leaf parity tests; imports adapted only.
"""Parity tests binding the streaming presence gate to the offline reference.

``backend/ownermic/inference/target_presence.py`` is the frame-synchronous
realtime twin of ``evaluation/presence_gate_sim.py`` (offline reference).
The backend must never import ``evaluation`` (it is not installable and is
only reachable through PYTHONPATH), so these tests are the single binding
contract between the two copies: every deterministic scenario, every grid
config, and every seeded random stream must make the **same open/close
decision on the same frame** in both implementations (exact equality, no
tolerance), and the streaming ``confidence`` must equal the reference
``smooth_identity`` output frame-for-frame.

These tests use only numpy and the standard library — no torch, no
hardware, no audio files — and run under both ``.venv`` and
``.venv-training`` (``tests.models.frozen_reference.gate`` itself imports nothing
beyond numpy and the standard library, verified before writing these tests).
"""

from __future__ import annotations

import unittest
from dataclasses import fields

import numpy as np

from modules.campplus.lib.presence import (
    PresenceGateConfig,
    PresenceGateError,
    StreamingPresenceGate,
)
from tests.models.frozen_reference.gate import (
    GateFrameFeatures,
    GateSimulationError,
    GateTuningConfig,
    simulate_gate,
    smooth_identity,
)

RANDOM_SEED = 20260907
RANDOM_FRAMES = 2000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _backend_to_eval(cfg: PresenceGateConfig) -> GateTuningConfig:
    """Translate a backend config into the offline ``GateTuningConfig``.

    Field-by-field (declared-order) translation: any drift between the two
    dataclasses (renamed/added/removed fields) surfaces here as a hard test
    failure instead of a silent semantic divergence.
    """
    return GateTuningConfig(**{f.name: getattr(cfg, f.name) for f in fields(GateTuningConfig)})


def _const(n: int, value: float) -> np.ndarray:
    return np.full(n, value, dtype=np.float64)


def _scenario(
    identity: np.ndarray,
    output_db: np.ndarray | float,
    veto: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Normalise one scenario to (identity, output_energy_db, veto-or-None)."""
    out = _const(len(identity), float(output_db)) if isinstance(output_db, float) else output_db
    return np.asarray(identity, dtype=np.float64), np.asarray(out, dtype=np.float64), veto


def _scenarios() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
    """Deterministic scenario battery (identity, output_db, veto-or-None).

    Each scenario deliberately exercises one corner of the state machine:
    all-zero / all-open / all-closed input, a single owner burst, scores
    oscillating inside the open/close hysteresis dead band, the fast-open
    path (score below open_threshold but >= fast_open_score with loud
    output, and its two negative controls), a veto that pulls down mid
    stream (plus the veto-disabled twin), flip attempts that probe the
    min_open/min_close lock boundaries, a short burst that falls right back,
    and oscillation across both thresholds.
    """
    n = 120
    single_burst = np.concatenate([_const(10, 0.05), _const(40, 0.8), _const(n - 50, 0.05)])
    band = np.concatenate([_const(40, 0.9), np.tile([0.44, 0.36], 45), _const(50, 0.05)])
    flip_attempts = np.concatenate(
        [_const(5, 0.8), _const(13, 0.1), _const(12, 0.8), _const(200, 0.1)]
    )
    short_run = np.concatenate([_const(6, 0.9), _const(94, 0.05)])
    oscillation = np.tile([0.9, 0.05], 100)
    veto_mid = np.concatenate([_const(100, 0.9), _const(80, 0.0)])
    return {
        "all_zero": _scenario(_const(64, 0.0), -60.0),
        "all_open": _scenario(_const(64, 1.0), 0.0),
        "all_closed": _scenario(_const(64, 0.02), 0.0),
        "single_burst": _scenario(single_burst, -60.0),
        "band_hysteresis": _scenario(band, -60.0),
        "fast_open_trigger": _scenario(_const(80, 0.50), 0.0),
        "fast_open_energy_low": _scenario(_const(80, 0.50), -60.0),
        "fast_open_score_low": _scenario(_const(80, 0.40), 0.0),
        "veto_midstream": _scenario(_const(180, 0.8), -60.0, veto_mid),
        "veto_channel_off": _scenario(_const(180, 0.8), -60.0),
        "flip_attempts": _scenario(flip_attempts, -60.0),
        "short_burst": _scenario(short_run, -60.0),
        "oscillation": _scenario(oscillation, -60.0),
    }


def _grid_configs() -> list[PresenceGateConfig]:
    """Four pairwise-distinct tuning combinations spanning all eight knobs."""
    return [
        # G1: shipped defaults.
        PresenceGateConfig(),
        # G2: identity-tracking EMA, short durations, permissive fast open.
        PresenceGateConfig(
            open_threshold=0.50,
            close_threshold=0.30,
            ema_open_alpha=1.0,
            ema_close_alpha=1.0,
            attack_frames=1,
            release_frames=2,
            min_open_frames=5,
            min_close_frames=4,
            fast_open_score=0.42,
            fast_open_energy_db=-20.0,
            veto_score=0.12,
        ),
        # G3: slow EMA, long locks, demanding fast open (needs >= -6 dB).
        PresenceGateConfig(
            open_threshold=0.60,
            close_threshold=0.25,
            ema_open_alpha=0.3,
            ema_close_alpha=0.05,
            attack_frames=3,
            release_frames=12,
            min_open_frames=40,
            min_close_frames=30,
            fast_open_score=0.50,
            fast_open_energy_db=-6.0,
            veto_score=0.10,
        ),
        # G4: fast_open_score == open_threshold (validation boundary) and a
        # veto threshold close under close_threshold.
        PresenceGateConfig(
            open_threshold=0.45,
            close_threshold=0.30,
            ema_open_alpha=0.7,
            ema_close_alpha=0.2,
            attack_frames=2,
            release_frames=4,
            min_open_frames=10,
            min_close_frames=8,
            fast_open_score=0.45,
            fast_open_energy_db=-12.0,
            veto_score=0.20,
        ),
    ]


def _build_eval_features(
    identity: np.ndarray,
    output_db: np.ndarray,
    veto: np.ndarray | None = None,
) -> GateFrameFeatures:
    n = len(identity)
    return GateFrameFeatures(
        identity_score=np.asarray(identity, dtype=np.float64),
        mixture_energy_db=np.zeros(n, dtype=np.float64),
        output_energy_db=np.asarray(output_db, dtype=np.float64),
        target_active=np.zeros(n, dtype=bool),
        identity_veto=None if veto is None else np.asarray(veto, dtype=np.float64),
    )


def _run_streaming(
    identity: np.ndarray,
    output_db: np.ndarray,
    cfg: PresenceGateConfig,
    veto: np.ndarray | None = None,
    *,
    frame_hz: int = 100,
) -> tuple[np.ndarray, np.ndarray, StreamingPresenceGate]:
    """Feed the whole stream to one fresh streaming gate, frame by frame."""
    gate = StreamingPresenceGate(cfg, frame_hz=frame_hz)
    n = len(identity)
    decisions = np.empty(n, dtype=bool)
    confidence = np.empty(n, dtype=np.float64)
    mixture = np.zeros(n, dtype=np.float64)
    for t in range(n):
        decisions[t] = gate.update(
            identity_score=float(identity[t]),
            mixture_energy_db=float(mixture[t]),
            output_energy_db=float(output_db[t]),
            identity_veto=None if veto is None else float(veto[t]),
        )
        confidence[t] = gate.confidence
    return decisions, confidence, gate


def _assert_parity(
    identity: np.ndarray,
    output_db: np.ndarray,
    cfg: PresenceGateConfig,
    veto: np.ndarray | None = None,
    *,
    label: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Assert exact frame-by-frame equality streaming vs offline reference."""
    feats = _build_eval_features(identity, output_db, veto)
    ref_gate = simulate_gate(feats, _backend_to_eval(cfg))
    ref_confidence = smooth_identity(feats, _backend_to_eval(cfg))
    stream_gate, stream_confidence, _ = _run_streaming(identity, output_db, cfg, veto)
    np.testing.assert_array_equal(
        ref_gate, stream_gate, err_msg=f"{label}: gate decision differs from reference"
    )
    np.testing.assert_array_equal(
        ref_confidence,
        stream_confidence,
        err_msg=f"{label}: streaming confidence differs from reference smooth_identity",
    )
    return ref_gate, stream_gate


def _open_runs(gate: np.ndarray) -> list[tuple[int, int]]:
    """(start, end_exclusive) of contiguous open runs."""
    runs: list[tuple[int, int]] = []
    n = len(gate)
    i = 0
    while i < n:
        if gate[i]:
            start = i
            while i < n and gate[i]:
                i += 1
            runs.append((start, i))
        else:
            i += 1
    return runs


def _random_sequences(
    rng: np.random.Generator,
    count: int,
    n_frames: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
    """Deterministic random scenario battery.

    Identity is a sequence of constant-level blocks (levels uniform in
    [0, 1]) so the asymmetric EMA repeatedly ramps up/down across the
    thresholds; output energy is uniform in [-50, -4] dB so the fast-open
    energy condition is met on some frames and missed on others.  Three of
    the five streams carry a veto channel, two leave it disabled (None).
    """
    sequences: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]] = []
    for seq_index in range(count):
        identity = np.empty(n_frames, dtype=np.float64)
        pos = 0
        while pos < n_frames:
            length = min(int(rng.integers(25, 175)), n_frames - pos)
            identity[pos : pos + length] = float(rng.uniform(0.0, 1.0))
            pos += length
        output_db = rng.uniform(-50.0, -4.0, size=n_frames)
        veto: np.ndarray | None = rng.uniform(0.0, 1.0, size=n_frames) if seq_index < 3 else None
        sequences.append((identity, output_db, veto))
    return sequences


# ---------------------------------------------------------------------------
# 1. Backend config validation
# ---------------------------------------------------------------------------


class TestPresenceGateConfigValidation(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = PresenceGateConfig()
        self.assertEqual(cfg.open_threshold, 0.55)
        self.assertEqual(cfg.close_threshold, 0.35)
        self.assertEqual(cfg.ema_open_alpha, 0.5)
        self.assertEqual(cfg.ema_close_alpha, 0.08)
        self.assertEqual(cfg.attack_frames, 1)
        self.assertEqual(cfg.release_frames, 20)
        self.assertEqual(cfg.min_open_frames, 30)
        self.assertEqual(cfg.min_close_frames, 20)
        self.assertEqual(cfg.fast_open_score, 0.45)
        self.assertEqual(cfg.fast_open_energy_db, -12.0)
        self.assertEqual(cfg.veto_score, 0.15)

    def test_invalid_configurations_raise(self) -> None:
        invalid: list[dict[str, object]] = [
            {"open_threshold": 0.3, "close_threshold": 0.5},
            {"open_threshold": 1.0},
            {"close_threshold": 0.0},
            {"ema_open_alpha": 0.0},
            {"ema_close_alpha": -0.1},
            {"ema_open_alpha": 1.5},
            {"attack_frames": -1},
            {"release_frames": -1},
            {"min_open_frames": -1},
            {"min_close_frames": -1},
            {"open_threshold": 0.4, "fast_open_score": 0.5},
            {"close_threshold": 0.1, "veto_score": 0.2},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(PresenceGateError):
                    PresenceGateConfig(**kwargs)  # type: ignore[arg-type]

    def test_validation_boundaries_accepted(self) -> None:
        PresenceGateConfig(open_threshold=0.5, fast_open_score=0.5)
        PresenceGateConfig(close_threshold=0.2, veto_score=0.2 - 1e-9)
        PresenceGateConfig(ema_open_alpha=1.0)
        PresenceGateConfig(attack_frames=0, release_frames=0, min_open_frames=0)

    def test_grid_configs_are_pairwise_distinct(self) -> None:
        grid = _grid_configs()
        self.assertGreaterEqual(len(grid), 4)
        keys = [
            (
                c.open_threshold,
                c.close_threshold,
                c.min_open_frames,
                c.min_close_frames,
                c.attack_frames,
                c.release_frames,
                c.fast_open_score,
                c.veto_score,
            )
            for c in grid
        ]
        self.assertEqual(len(keys), len(set(keys)))


# ---------------------------------------------------------------------------
# 2. Config contract parity with GateTuningConfig
# ---------------------------------------------------------------------------


class TestConfigContractParity(unittest.TestCase):
    def test_field_names_and_defaults_match_reference(self) -> None:
        """Both dataclasses declare the same fields with the same defaults."""
        pairs = [(f.name, f.default) for f in fields(PresenceGateConfig)]
        reference_pairs = [(f.name, f.default) for f in fields(GateTuningConfig)]
        self.assertEqual(pairs, reference_pairs)

    def test_invalid_inputs_raise_on_both_sides(self) -> None:
        """Backend validation must reject exactly what the reference rejects."""
        invalid: list[dict[str, object]] = [
            {"open_threshold": 0.3, "close_threshold": 0.5},
            {"open_threshold": 1.0},
            {"close_threshold": 0.0},
            {"ema_open_alpha": 0.0},
            {"ema_close_alpha": -0.1},
            {"ema_open_alpha": 1.5},
            {"attack_frames": -1},
            {"release_frames": -1},
            {"min_open_frames": -1},
            {"min_close_frames": -1},
            {"open_threshold": 0.4, "fast_open_score": 0.5},
            {"close_threshold": 0.1, "veto_score": 0.2},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(GateSimulationError):
                    GateTuningConfig(**kwargs)  # type: ignore[arg-type]
                with self.assertRaises(PresenceGateError):
                    PresenceGateConfig(**kwargs)  # type: ignore[arg-type]

    def test_backend_to_eval_round_trip(self) -> None:
        for cfg in _grid_configs():
            translated = _backend_to_eval(cfg)
            for field in fields(GateTuningConfig):
                self.assertEqual(
                    getattr(cfg, field.name),
                    getattr(translated, field.name),
                    msg=f"field {field.name} diverged for {cfg}",
                )


# ---------------------------------------------------------------------------
# 3. Streaming gate basics
# ---------------------------------------------------------------------------


class TestStreamingGateBasics(unittest.TestCase):
    def test_initial_state(self) -> None:
        cfg = _grid_configs()[0]
        gate = StreamingPresenceGate(cfg, frame_hz=100)
        self.assertFalse(gate.open)
        self.assertIsNone(gate.confidence)
        self.assertEqual(gate.frame_hz, 100)
        self.assertIs(gate.config, cfg)

    def test_update_returns_current_open_state(self) -> None:
        identity, output_db, _ = _scenarios()["all_open"]
        cfg = _grid_configs()[0]
        gate = StreamingPresenceGate(cfg, frame_hz=100)
        for t in range(len(identity)):
            opened = gate.update(
                identity_score=float(identity[t]),
                mixture_energy_db=0.0,
                output_energy_db=float(output_db[t]),
            )
            self.assertIs(opened, gate.open)
        self.assertTrue(gate.open)

    def test_confidence_tracks_smoothed_identity(self) -> None:
        identity, output_db, _ = _scenarios()["single_burst"]
        gate = StreamingPresenceGate(_grid_configs()[0], frame_hz=100)
        gate.update(identity_score=0.9, mixture_energy_db=0.0, output_energy_db=0.0)
        self.assertAlmostEqual(gate.confidence, 0.9, delta=1e-12)
        gate.update(identity_score=0.1, mixture_energy_db=0.0, output_energy_db=0.0)
        expected = 0.08 * 0.1 + 0.92 * 0.9  # falling: ema_close_alpha = 0.08
        self.assertAlmostEqual(gate.confidence, expected, delta=1e-12)

    def test_frame_hz_must_be_positive(self) -> None:
        cfg = _grid_configs()[0]
        with self.assertRaises(ValueError):
            StreamingPresenceGate(cfg, frame_hz=0)
        with self.assertRaises(ValueError):
            StreamingPresenceGate(cfg, frame_hz=-100)

    def test_reset_restores_initial_condition(self) -> None:
        identity, output_db, _ = _scenarios()["all_open"]
        gate = StreamingPresenceGate(_grid_configs()[0], frame_hz=100)
        for t in range(10):
            gate.update(
                identity_score=float(identity[t]),
                mixture_energy_db=0.0,
                output_energy_db=float(output_db[t]),
            )
        self.assertTrue(gate.open)
        self.assertIsNotNone(gate.confidence)
        gate.reset()
        self.assertFalse(gate.open)
        self.assertIsNone(gate.confidence)

    def test_state_stays_scalar_after_many_updates(self) -> None:
        """Worker-thread contract: no arrays/buffers grow inside the gate."""
        identity, output_db, veto = _random_sequences(
            np.random.default_rng(RANDOM_SEED), 1, RANDOM_FRAMES
        )[0]
        cfg = _grid_configs()[0]
        gate = StreamingPresenceGate(cfg, frame_hz=100)
        for t in range(RANDOM_FRAMES):
            gate.update(
                identity_score=float(identity[t]),
                mixture_energy_db=0.0,
                output_energy_db=float(output_db[t]),
                identity_veto=None if veto is None else float(veto[t]),
            )
        scalar = (bool, int, float, type(None))
        for name in StreamingPresenceGate.__slots__:
            value = getattr(gate, name)
            if name in ("_config", "_frame_hz"):
                continue
            self.assertIsInstance(value, scalar, msg=f"state slot {name} is not scalar")


# ---------------------------------------------------------------------------
# 4. Deterministic scenario parity across the config grid
# ---------------------------------------------------------------------------


class TestDeterministicScenarioParity(unittest.TestCase):
    def test_all_scenarios_all_grid_configs_frame_equal(self) -> None:
        """Every scenario x grid config must match simulate_gate frame-for-frame."""
        scenarios = _scenarios()
        grid = _grid_configs()
        self.assertGreaterEqual(len(scenarios), 10)
        for name, (identity, output_db, veto) in scenarios.items():
            for cfg_index, cfg in enumerate(grid):
                with self.subTest(scenario=name, cfg=cfg_index):
                    _assert_parity(
                        identity,
                        output_db,
                        cfg,
                        veto,
                        label=f"scenario={name}, cfg=G{cfg_index + 1}",
                    )


# ---------------------------------------------------------------------------
# 5. Seeded random parity (>= 2000 frames, 5 sequences x 3 configs)
# ---------------------------------------------------------------------------


class TestSeededRandomParity(unittest.TestCase):
    def test_random_streams_frame_equal(self) -> None:
        rng = np.random.default_rng(RANDOM_SEED)
        sequences = _random_sequences(rng, 5, RANDOM_FRAMES)
        grid = _grid_configs()[:3]
        references: dict[tuple[int, int], np.ndarray] = {}
        for seq_index, (identity, output_db, veto) in enumerate(sequences):
            self.assertGreaterEqual(len(identity), RANDOM_FRAMES)
            for cfg_index, cfg in enumerate(grid):
                with self.subTest(seq=seq_index, cfg=cfg_index):
                    ref_gate, _ = _assert_parity(
                        identity,
                        output_db,
                        cfg,
                        veto,
                        label=f"random seq={seq_index}, cfg=G{cfg_index + 1}",
                    )
                    references[(seq_index, cfg_index)] = ref_gate
        # The fixed seed must actually exercise both open and closed frames
        # for the default config (otherwise the parity check is vacuous).
        self.assertTrue(bool(references[(0, 0)].any()))
        self.assertFalse(bool(references[(0, 0)].all()))
        # Re-generating with the same seed reproduces the identical streams.
        again = _random_sequences(np.random.default_rng(RANDOM_SEED), 5, RANDOM_FRAMES)
        for (identity, output_db, veto), (i2, o2, v2) in zip(sequences, again, strict=True):
            np.testing.assert_array_equal(identity, i2)
            np.testing.assert_array_equal(output_db, o2)
            if veto is None:
                self.assertIsNone(v2)
            else:
                np.testing.assert_array_equal(veto, v2)


# ---------------------------------------------------------------------------
# 6. Semantic lockstep (exact expectations, verified against the reference)
# ---------------------------------------------------------------------------


class TestSemanticLockstep(unittest.TestCase):
    """Exact expected behaviour on default and corner configs.

    Expected run boundaries below were derived from the offline reference
    (``simulate_gate`` is the oracle the parity contract locks to) and
    re-derived by hand where noted; each test also re-asserts streaming ==
    reference equality so the numbers cannot drift one-sidedly.
    """

    def test_high_score_opens_on_first_frame(self) -> None:
        cfg = _grid_configs()[0]
        identity = _const(60, 0.9)
        ref, stream = _assert_parity(identity, _const(60, -60.0), cfg)
        # The initial closed period is seeded with min_close_frames + 1, so
        # frame 0 opens immediately and never closes while the score holds.
        self.assertEqual(_open_runs(ref), [(0, 60)])
        self.assertEqual(_open_runs(stream), [(0, 60)])

    def test_single_burst_first_open_and_release(self) -> None:
        cfg = _grid_configs()[0]
        identity = np.concatenate([_const(10, 0.05), _const(40, 0.8), _const(70, 0.05)])
        ref, _ = _assert_parity(identity, _const(120, -60.0), cfg)
        # Open needs the identity EMA (ema_open_alpha=0.5) to clear 0.55:
        # frame 10 seeds the ramp (0.425), frame 11 opens.  After the burst
        # the slow close EMA (0.08) decays below close_threshold and the 20
        # release frames accumulate; run = [11, 79).
        self.assertEqual(_open_runs(ref), [(11, 79)])

    def test_short_burst_holds_open_for_default_locks(self) -> None:
        """Open followed by an immediate drop: min_open + release dominate."""
        cfg = _grid_configs()[0]
        identity = np.concatenate([_const(6, 0.9), _const(94, 0.05)])
        ref, _ = _assert_parity(identity, _const(100, -60.0), cfg)
        # Gate opens at frame 0, the score dies at frame 6, but the gate may
        # only release once consecutive_open > min_open_frames (30) and then
        # needs release_frames (20) consecutive close candidates: run =
        # [0, 49) = min_open + release - 1 frames.
        self.assertEqual(_open_runs(ref), [(0, 49)])

    def test_short_burst_exact_min_open_run(self) -> None:
        """release_frames=1: the open run length equals min_open exactly."""
        cfg = PresenceGateConfig(
            open_threshold=0.55,
            close_threshold=0.35,
            ema_open_alpha=1.0,
            ema_close_alpha=1.0,
            attack_frames=1,
            release_frames=1,
            min_open_frames=30,
            min_close_frames=1,
            fast_open_score=0.5,
            fast_open_energy_db=-100.0,
            veto_score=0.15,
        )
        identity = np.concatenate([_const(3, 0.9), _const(97, 0.05)])
        ref, _ = _assert_parity(identity, _const(100, -60.0), cfg)
        # Score died at frame 3, but min_open_frames=30 is always respected:
        # the first eligible close frame (consecutive_open > 30) is frame 30,
        # so gate[0..29] stay open.  This is the flip-frame counting contract:
        # consecutive_open starts at 1 on the opening frame itself.
        self.assertEqual(_open_runs(ref), [(0, 30)])

    def test_min_open_min_close_flip_boundaries(self) -> None:
        """Exact lock boundaries around both flips (hand-derived, verified)."""
        cfg = PresenceGateConfig(
            open_threshold=0.55,
            close_threshold=0.35,
            ema_open_alpha=1.0,
            ema_close_alpha=1.0,
            attack_frames=1,
            release_frames=1,
            min_open_frames=10,
            min_close_frames=8,
            fast_open_score=0.5,
            fast_open_energy_db=-100.0,
            veto_score=0.15,
        )
        identity = np.concatenate([_const(5, 0.8), _const(13, 0.1), _const(20, 0.8)])
        ref, stream = _assert_parity(identity, _const(38, -60.0), cfg)
        runs = [(0, 10), (18, 38)]
        self.assertEqual(_open_runs(ref), runs)
        self.assertEqual(_open_runs(stream), runs)
        # Derivation: open at frame 0 (seeded initial closed period); close
        # candidates start at frame 5 (score 0.1) but the gate may release
        # only when consecutive_open > 10 -> close at frame 10, so the open
        # run is exactly min_open_frames.  The close flip seeds
        # consecutive_closed = 1; reopen needs consecutive_closed > 8 ->
        # frame 18, so the closed run is exactly min_close_frames.  Frame 10
        # (the closing frame) and frame 18 (the reopening frame) both report
        # their new state, matching gate[t] semantics.
        self.assertFalse(ref[10])
        self.assertTrue(ref[18])

    def test_attack_frames_accumulate_and_reset(self) -> None:
        cfg = PresenceGateConfig(
            open_threshold=0.55,
            close_threshold=0.35,
            ema_open_alpha=1.0,
            ema_close_alpha=1.0,
            attack_frames=3,
            release_frames=1,
            min_open_frames=1,
            min_close_frames=1,
            fast_open_score=0.5,
            fast_open_energy_db=-100.0,
            veto_score=0.15,
        )
        identity = _const(40, 0.8)
        ref, _ = _assert_parity(identity, _const(40, -60.0), cfg)
        self.assertEqual(_open_runs(ref), [(2, 40)])
        # A single sub-attack frame resets the counter: 0.8, 0.8, 0.1, then
        # three clean frames -> open at frame 5.
        identity_reset = np.concatenate([_const(2, 0.8), _const(1, 0.1), _const(40, 0.8)])
        ref2, _ = _assert_parity(identity_reset, _const(43, -60.0), cfg)
        self.assertEqual(_open_runs(ref2), [(5, 43)])

    def test_veto_close_respects_min_open(self) -> None:
        cfg = PresenceGateConfig(
            open_threshold=0.55,
            close_threshold=0.35,
            ema_open_alpha=1.0,
            ema_close_alpha=1.0,
            attack_frames=1,
            release_frames=1,
            min_open_frames=50,
            min_close_frames=1,
            fast_open_score=0.5,
            fast_open_energy_db=-100.0,
            veto_score=0.15,
        )
        identity = _const(80, 0.8)
        veto = np.concatenate([_const(10, 0.9), _const(70, 0.0)])
        ref, _ = _assert_parity(identity, _const(80, -60.0), cfg, veto)
        # Veto becomes active at frame 10 (EMA alpha 1.0 -> raw 0.0 < 0.15)
        # while identity stays 0.8, yet min_open_frames=50 is respected: the
        # first eligible close is frame 50 -> run [0, 50).
        self.assertEqual(_open_runs(ref), [(0, 50)])

    def test_veto_forced_close_waives_min_close(self) -> None:
        cfg = PresenceGateConfig(
            open_threshold=0.55,
            close_threshold=0.35,
            ema_open_alpha=1.0,
            ema_close_alpha=1.0,
            attack_frames=1,
            release_frames=1,
            min_open_frames=2,
            min_close_frames=30,
            fast_open_score=0.5,
            fast_open_energy_db=-100.0,
            veto_score=0.15,
        )
        identity = _const(80, 0.8)
        veto = np.concatenate([_const(10, 0.9), _const(8, 0.0), _const(62, 0.9)])
        ref, _ = _assert_parity(identity, _const(80, -60.0), cfg, veto)
        # Veto-active frames 10..17 force a close at frame 10 with
        # closed_due_to_veto=True; reopening therefore does not wait out
        # min_close_frames=30 and happens on the first clean frame, 18.
        self.assertEqual(_open_runs(ref), [(0, 10), (18, 80)])

    def test_veto_midstream_pulls_down_high_identity(self) -> None:
        cfg = PresenceGateConfig(
            open_threshold=0.55,
            close_threshold=0.35,
            ema_open_alpha=1.0,
            ema_close_alpha=1.0,
            attack_frames=1,
            release_frames=1,
            min_open_frames=2,
            min_close_frames=1,
            fast_open_score=0.5,
            fast_open_energy_db=-100.0,
            veto_score=0.15,
        )
        identity = _const(180, 0.8)
        veto = np.concatenate([_const(100, 0.9), _const(80, 0.0)])
        ref, _ = _assert_parity(identity, _const(180, -60.0), cfg, veto)
        # Identity never drops, but veto-active from frame 100 wins over the
        # open candidate (even the fast path) and closes the gate at 100.
        self.assertEqual(_open_runs(ref), [(0, 100)])
        self.assertFalse(bool(ref[100:].any()))

    def test_veto_disabled_channel_never_acts(self) -> None:
        """identity_veto=None every frame == reference with no veto array."""
        cfg = PresenceGateConfig(
            open_threshold=0.55,
            close_threshold=0.35,
            ema_open_alpha=1.0,
            ema_close_alpha=1.0,
            attack_frames=1,
            release_frames=1,
            min_open_frames=2,
            min_close_frames=1,
            fast_open_score=0.5,
            fast_open_energy_db=-100.0,
            veto_score=0.15,
        )
        identity = _const(180, 0.8)
        ref, _ = _assert_parity(identity, _const(180, -60.0), cfg, None)
        self.assertEqual(_open_runs(ref), [(0, 180)])
        self.assertTrue(bool(ref.all()))

    def test_hysteresis_band_oscillation_holds_state(self) -> None:
        cfg = _grid_configs()[0]
        band = np.concatenate([_const(40, 0.9), np.tile([0.44, 0.36], 45), _const(50, 0.05)])
        ref, _ = _assert_parity(band, _const(180, -60.0), cfg)
        # 0.44/0.36 oscillate inside the (0.35, 0.55) dead band: no close
        # candidate forms and the gate stays open through the whole band
        # (frames 40..129); it only releases after the 0.05 tail drags the
        # slow close EMA under 0.35 and 20 release frames elapse.
        self.assertTrue(bool(ref[:130].all()))
        self.assertEqual(_open_runs(ref), [(0, 151)])
        self.assertFalse(bool(ref[151:].any()))

    def test_exact_threshold_inclusive_comparisons(self) -> None:
        cfg = PresenceGateConfig(
            open_threshold=0.55,
            close_threshold=0.35,
            ema_open_alpha=1.0,
            ema_close_alpha=1.0,
            attack_frames=1,
            release_frames=1,
            min_open_frames=1,
            min_close_frames=1,
            fast_open_score=0.5,
            fast_open_energy_db=-100.0,
            veto_score=0.15,
        )
        identity = np.concatenate([_const(10, 0.55), _const(30, 0.35)])
        ref, _ = _assert_parity(identity, _const(40, -60.0), cfg)
        # smoothed >= open_threshold is inclusive (0.55 opens frame 0) and
        # smoothed <= close_threshold is inclusive (the first exact-0.35
        # frame closes immediately, min_open_frames=1).
        self.assertEqual(_open_runs(ref), [(0, 10)])

    def test_fast_open_path_below_open_threshold(self) -> None:
        cfg = PresenceGateConfig(
            open_threshold=0.55,
            close_threshold=0.35,
            ema_open_alpha=0.02,  # smoothed never approaches 0.55
            ema_close_alpha=0.02,
            attack_frames=1,
            release_frames=1,
            min_open_frames=1,
            min_close_frames=1,
            fast_open_score=0.45,
            fast_open_energy_db=-12.0,
            veto_score=0.15,
        )
        identity = _const(60, 0.50)  # >= fast_open_score, < open_threshold
        ref, _ = _assert_parity(identity, _const(60, 0.0), cfg)
        self.assertEqual(_open_runs(ref), [(0, 60)])
        # Negative controls: same score but quiet output, or output loud but
        # score below fast_open_score -> the gate never opens.
        ref_low_energy, _ = _assert_parity(identity, _const(60, -60.0), cfg)
        self.assertEqual(_open_runs(ref_low_energy), [])
        ref_low_score, _ = _assert_parity(_const(60, 0.40), _const(60, 0.0), cfg)
        self.assertEqual(_open_runs(ref_low_score), [])

    def test_veto_blocks_fast_open(self) -> None:
        cfg = PresenceGateConfig(
            open_threshold=0.55,
            close_threshold=0.35,
            ema_open_alpha=0.02,
            ema_close_alpha=1.0,
            attack_frames=1,
            release_frames=1,
            min_open_frames=1,
            min_close_frames=1,
            fast_open_score=0.45,
            fast_open_energy_db=-12.0,
            veto_score=0.15,
        )
        identity = _const(60, 0.50)
        veto = _const(60, 0.0)  # veto-active from the first frame
        ref, _ = _assert_parity(identity, _const(60, 0.0), cfg, veto)
        # Fast-open condition holds but the veto overrides the open
        # candidate, so the gate stays closed the whole stream.
        self.assertEqual(_open_runs(ref), [])

    def test_oscillation_across_thresholds_default_release(self) -> None:
        cfg = _grid_configs()[0]
        oscillation = np.tile([0.9, 0.05], 100)
        ref, _ = _assert_parity(oscillation, _const(200, -60.0), cfg)
        # Close candidates appear only on the odd (0.05) frames, so the 20
        # release frames never accumulate consecutively; the gate opens at
        # frame 0 and stays open (min_open lock holds in between).
        self.assertEqual(_open_runs(ref), [(0, 200)])


# ---------------------------------------------------------------------------
# 7. Prefix invariance: the streaming gate must not see the future
# ---------------------------------------------------------------------------


class TestPrefixInvariance(unittest.TestCase):
    """Outputs for the first N frames must not depend on later frames."""

    def _check_prefix(
        self,
        identity: np.ndarray,
        output_db: np.ndarray,
        cfg: PresenceGateConfig,
        veto: np.ndarray | None,
        prefix_lengths: list[int],
        label: str,
    ) -> None:
        full_decisions, full_confidence, _ = _run_streaming(identity, output_db, cfg, veto)
        for n in prefix_lengths:
            with self.subTest(label=label, n=n):
                prefix_decisions, prefix_confidence, _ = _run_streaming(
                    identity[:n], output_db[:n], cfg, None if veto is None else veto[:n]
                )
                np.testing.assert_array_equal(
                    full_decisions[:n], prefix_decisions, err_msg=f"{label}: N={n} decisions"
                )
                np.testing.assert_array_equal(
                    full_confidence[:n],
                    prefix_confidence,
                    err_msg=f"{label}: N={n} confidence",
                )

    def test_single_burst_prefixes_including_the_opening_frame(self) -> None:
        """N spans the frame where the reference first opens (frame 11)."""
        cfg = _grid_configs()[0]
        identity = np.concatenate([_const(10, 0.05), _const(40, 0.8), _const(70, 0.05)])
        output_db = _const(120, -60.0)
        feats = _build_eval_features(identity, output_db)
        ref = simulate_gate(feats, _backend_to_eval(cfg))
        first_open = int(np.argmax(ref))
        self.assertEqual(first_open, 11)
        self._check_prefix(
            identity,
            output_db,
            cfg,
            None,
            [first_open, first_open + 1, first_open + 20, len(identity) // 2, len(identity)],
            "single_burst",
        )

    def test_random_prefixes_at_reference_open_frames(self) -> None:
        rng = np.random.default_rng(RANDOM_SEED)
        identity, output_db, veto = _random_sequences(rng, 5, RANDOM_FRAMES)[0]
        cfg = _grid_configs()[0]
        feats = _build_eval_features(identity, output_db, veto)
        ref = simulate_gate(feats, _backend_to_eval(cfg))
        first_open = int(np.argmax(ref))
        if first_open == 0:
            self.skipTest("random stream opens on frame 0; use mid-stream prefix instead")
        self._check_prefix(
            identity,
            output_db,
            cfg,
            veto,
            [first_open, first_open + 1, 700],
            "random_seq0",
        )

    def test_veto_stream_prefix_before_and_after_veto_onset(self) -> None:
        cfg = PresenceGateConfig(
            open_threshold=0.55,
            close_threshold=0.35,
            ema_open_alpha=1.0,
            ema_close_alpha=1.0,
            attack_frames=1,
            release_frames=1,
            min_open_frames=2,
            min_close_frames=1,
            fast_open_score=0.5,
            fast_open_energy_db=-100.0,
            veto_score=0.15,
        )
        identity = _const(180, 0.8)
        veto = np.concatenate([_const(100, 0.9), _const(80, 0.0)])
        output_db = _const(180, -60.0)
        self._check_prefix(identity, output_db, cfg, veto, [60, 100, 101, 150], "veto_mid")


# ---------------------------------------------------------------------------
# 8. Reset replay and instance isolation
# ---------------------------------------------------------------------------


class TestResetAndIsolation(unittest.TestCase):
    def _burst_features(self) -> tuple[np.ndarray, np.ndarray]:
        identity = np.concatenate([_const(10, 0.05), _const(40, 0.8), _const(70, 0.05)])
        return identity, _const(120, -60.0)

    def test_reset_replays_identical_stream(self) -> None:
        identity, output_db = self._burst_features()
        cfg = _grid_configs()[1]
        gate = StreamingPresenceGate(cfg, frame_hz=100)
        first, first_conf, _ = _run_streaming(identity, output_db, cfg)
        gate.reset()
        second = np.empty(len(identity), dtype=bool)
        second_conf = np.empty(len(identity), dtype=np.float64)
        for t in range(len(identity)):
            second[t] = gate.update(
                identity_score=float(identity[t]),
                mixture_energy_db=0.0,
                output_energy_db=float(output_db[t]),
            )
            second_conf[t] = gate.confidence
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first_conf, second_conf)
        self.assertEqual(first_conf[-1], second_conf[-1])

    def test_interleaved_instances_do_not_interfere(self) -> None:
        """Two gates fed the same stream alternately decide identically."""
        identity, output_db = self._burst_features()
        cfg = _grid_configs()[1]
        gate_a = StreamingPresenceGate(cfg, frame_hz=100)
        gate_b = StreamingPresenceGate(cfg, frame_hz=100)
        out_a = np.empty(len(identity), dtype=bool)
        out_b = np.empty(len(identity), dtype=bool)
        for t in range(len(identity)):
            out_a[t] = gate_a.update(
                identity_score=float(identity[t]),
                mixture_energy_db=0.0,
                output_energy_db=float(output_db[t]),
            )
            out_b[t] = gate_b.update(
                identity_score=float(identity[t]),
                mixture_energy_db=0.0,
                output_energy_db=float(output_db[t]),
            )
        solo, _, _ = _run_streaming(identity, output_db, cfg)
        np.testing.assert_array_equal(out_a, out_b)
        np.testing.assert_array_equal(solo, out_a)
        self.assertEqual(gate_a.confidence, gate_b.confidence)

    def test_fresh_instance_starts_clean_regardless_of_other_gates(self) -> None:
        identity, output_db = self._burst_features()
        cfg = _grid_configs()[1]
        gate_a = StreamingPresenceGate(cfg, frame_hz=100)
        prefix_a = np.empty(60, dtype=bool)
        for t in range(60):
            prefix_a[t] = gate_a.update(
                identity_score=float(identity[t]),
                mixture_energy_db=0.0,
                output_energy_db=float(output_db[t]),
            )
        # A new gate created after A was already deep into the stream must
        # behave exactly like a fresh gate on the full stream.
        gate_b = StreamingPresenceGate(cfg, frame_hz=100)
        solo, _, _ = _run_streaming(identity, output_db, cfg)
        fresh = np.empty(len(identity), dtype=bool)
        for t in range(len(identity)):
            fresh[t] = gate_b.update(
                identity_score=float(identity[t]),
                mixture_energy_db=0.0,
                output_energy_db=float(output_db[t]),
            )
        np.testing.assert_array_equal(solo, fresh)
        # And A's prefix through frame 60 agrees with the clean run.
        np.testing.assert_array_equal(solo[:60], prefix_a)

    def test_no_class_level_mutable_state(self) -> None:
        """Guard against accidentally shared mutable defaults between gates."""
        for name, value in vars(StreamingPresenceGate).items():
            if name.startswith("__") and name.endswith("__"):
                continue
            self.assertNotIsInstance(value, (list, dict, set, np.ndarray))


if __name__ == "__main__":
    unittest.main()
