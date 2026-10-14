# Frozen leaf tests; old controller integration excluded. Imports adapted only.
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from modules.campplus.lib.identity_seam import IdentityObservation
from modules.campplus.lib.identity_stream import (
    IdentityStreamError,
    SlidingIdentityProvider,
    WindowEmbeddingExtractor,
)
from tests.models.frozen_reference.identity import (
    IdentityStreamConfig,
    sliding_identity,
)
from tests.models.frozen_reference.identity import (
    frame_energy_db as offline_frame_energy_db,
)

SAMPLE_RATE = 16000
FRAME = 160
SILENCE_FLOOR_DBFS = -55.0


def _unit_vector(dim: int = 192, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.normal(0.0, 1.0, dim).astype(np.float32)
    return np.ascontiguousarray(vec / float(np.linalg.norm(vec)), dtype=np.float32)


class _FakeEmbedder:
    """Pure, deterministic function of the window content.

    Peak amplitude selects the outcome so streams can be scripted by simple
    signal styles: > 0.95 -> embed failure (``None``); [0.7, 0.95) -> cosine
    -0.8 (negative); [0.55, 0.7) -> cosine 0.0; below -> cosine 0.85.
    """

    def __init__(self, target: np.ndarray) -> None:
        self.target = np.ascontiguousarray(np.asarray(target, dtype=np.float32).reshape(-1))
        dim = self.target.shape[0]
        basis = np.eye(dim, dtype=np.float32)[0]
        orth = basis - float(np.dot(basis, self.target)) * self.target
        orth_norm = float(np.linalg.norm(orth))
        self._orth = np.ascontiguousarray(orth / orth_norm, dtype=np.float32)
        self.calls = 0

    def embed(self, audio: np.ndarray) -> np.ndarray | None:
        self.calls += 1
        window = np.asarray(audio, dtype=np.float32).reshape(-1)
        peak = float(np.abs(window).max()) if window.size else 0.0
        if peak > 0.95:
            return None
        if peak >= 0.70:
            cosine = -0.8
        elif peak >= 0.55:
            cosine = 0.0
        else:
            cosine = 0.85
        sine = float(np.sqrt(max(0.0, 1.0 - cosine * cosine)))
        vector = cosine * self.target + sine * self._orth
        norm = float(np.linalg.norm(vector))
        return np.ascontiguousarray(vector / norm, dtype=np.float32)


def _square(amplitude: float, samples: int) -> np.ndarray:
    """Deterministic square wave whose peak amplitude is *amplitude* exactly."""
    half = FRAME // 2
    pattern = np.array([amplitude] * half + [-amplitude] * half, dtype=np.float32)
    return np.tile(pattern, (samples + FRAME - 1) // FRAME)[:samples]


def _style_block(style: str, frames: int, rng: np.random.Generator) -> np.ndarray:
    """One audio block of *frames* 10 ms frames for a named content style."""
    if style == "quiet":  # above the silence floor, embed cosine +0.85
        return rng.normal(0.0, 0.02, frames * FRAME).astype(np.float32)
    if style == "silence":  # pure zeros: window RMS == -200 dBFS < floor
        return np.zeros(frames * FRAME, dtype=np.float32)
    if style == "medium":  # peak 0.8 -> embed returns a negative-cosine vector
        return _square(0.8, frames * FRAME)
    if style == "zero-cos":  # peak 0.6 -> embed returns an orthogonal vector
        return _square(0.6, frames * FRAME)
    if style == "loud":  # peak 1.0 -> embed returns None
        return _square(1.0, frames * FRAME)
    raise ValueError(f"unknown style {style!r}")


def _profile_audio(total_frames: int, seed: int = 42) -> np.ndarray:
    """Style profile (defaults to 20 s = 2000 frames) truncated to length.

    Zone lengths all exceed the largest window (1000 ms = 100 frames) plus a
    hop, so every config sees fully-contained silence / loud / medium zones.
    """
    rng = np.random.default_rng(seed)
    styles = [
        ("quiet", 240),
        ("medium", 220),
        ("silence", 260),
        ("loud", 240),
        ("zero-cos", 200),
        ("quiet", 220),
        ("silence", 220),
        ("medium", 240),
        ("loud", 160),
    ]
    blocks: list[np.ndarray] = []
    remaining = total_frames
    while remaining > 0:
        for style, length in styles:
            if remaining == 0:
                break
            take = min(length, remaining)
            blocks.append(_style_block(style, take, rng))
            remaining -= take
    return np.ascontiguousarray(np.concatenate(blocks), dtype=np.float32)


def _offline_scores(
    audio: np.ndarray, enrollment: np.ndarray, fake: _FakeEmbedder, window_ms: int, hop_ms: int
) -> np.ndarray:
    """Offline per-frame identity scores for *audio* (float64, reference)."""
    config = IdentityStreamConfig(
        sample_rate=SAMPLE_RATE,
        frame_samples=FRAME,
        window_ms=window_ms,
        hop_ms=hop_ms,
        silence_floor_dbfs=SILENCE_FLOOR_DBFS,
        num_threads=1,
    )
    return np.asarray(
        sliding_identity(audio, enrollment, config, embed=fake.embed), dtype=np.float64
    )


def _feed(provider: SlidingIdentityProvider, audio: np.ndarray) -> list[IdentityObservation]:
    """Feed *audio* frame by frame (10 ms chunks); every observation is non-None."""
    observations: list[IdentityObservation] = []
    frames = audio.shape[0] // FRAME
    for t in range(frames):
        chunk = audio[t * FRAME : (t + 1) * FRAME]
        observation = provider.observe(chunk, sample_rate=SAMPLE_RATE)
        if observation is None:
            raise AssertionError(f"observe() returned None at frame {t}")
        observations.append(observation)
    return observations


def _to_arrays(
    observations: list[IdentityObservation],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    scores = np.asarray([float(obs.score) for obs in observations], dtype=np.float64)
    vetoes: np.ndarray | None = None
    if observations and observations[0].veto is not None:
        vetoes = np.asarray([float(obs.veto) for obs in observations], dtype=np.float64)
    energy = np.asarray([float(obs.mixture_energy_db) for obs in observations], dtype=np.float64)
    return scores, vetoes, energy


class TestFrameExactEquivalence(unittest.TestCase):
    """Streaming provider vs offline ``sliding_identity``, bitwise equal."""

    def _assert_equivalence(
        self,
        window_ms: int,
        hop_ms: int,
        veto_window_ms: int | None,
        total_frames: int = 2000,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, SlidingIdentityProvider, list[IdentityObservation]]:
        audio = _profile_audio(total_frames, seed=seed)
        enrollment = _unit_vector()
        fake = _FakeEmbedder(enrollment)

        provider = SlidingIdentityProvider(
            enrollment,
            fake,
            window_ms=window_ms,
            hop_ms=hop_ms,
            silence_floor_dbfs=SILENCE_FLOOR_DBFS,
            veto_window_ms=veto_window_ms,
        )
        provider.initialize()
        observations = _feed(provider, audio)
        scores, vetoes, energy = _to_arrays(observations)

        # Frame-exact identity scores.
        ref_scores = _offline_scores(audio, enrollment, fake, window_ms, hop_ms)
        np.testing.assert_array_equal(scores, ref_scores)

        # Frame-exact mixture energy (same per-frame formula as offline).
        ref_energy = offline_frame_energy_db(audio, FRAME)
        np.testing.assert_array_equal(energy, ref_energy)

        if veto_window_ms is None:
            self.assertTrue(all(obs.veto is None for obs in observations))
        else:
            self.assertIsNotNone(vetoes)
            ref_veto = _offline_scores(audio, enrollment, fake, veto_window_ms, hop_ms)
            np.testing.assert_array_equal(vetoes, ref_veto)
        return scores, ref_scores, provider, observations

    def test_config_600_100_full_20s_stream(self) -> None:
        """Default 600 ms / 100 ms window over a full 20 s (2000 frame) stream."""
        scores, ref, _, _ = self._assert_equivalence(600, 100, 600)
        self.assertEqual(scores.shape[0], 2000)
        np.testing.assert_array_equal(scores, ref)

    def test_config_600_100_leading_frames_and_reset_replay(self) -> None:
        """Frames before the first window hold 0.0; reset() replays identically."""
        scores, _, provider, _ = self._assert_equivalence(600, 100, 600)
        first_window = 60  # frames before 600 ms have no complete window
        self.assertTrue(np.all(scores[: first_window - 1] == 0.0))
        self.assertGreater(scores[first_window], 0.0)
        provider.reset()
        observations = _feed(provider, _profile_audio(2000))
        replay, _, replay_energy = _to_arrays(observations)
        np.testing.assert_array_equal(replay, scores)
        self.assertEqual(replay_energy.shape[0], scores.shape[0])
        self.assertEqual(provider.algorithmic_latency_ms, 700.0)

    def test_config_400_100_with_veto_600(self) -> None:
        """Main window 400 ms plus an independent 600 ms veto window."""
        _, _, provider, _ = self._assert_equivalence(400, 100, 600)
        self.assertEqual(provider.window_ms, 400)
        self.assertEqual(provider.veto_window_ms, 600)

    def test_config_1000_200_no_veto(self) -> None:
        """Window 1000 ms / hop 200 ms with the veto channel disabled."""
        _, _, provider, _ = self._assert_equivalence(1000, 200, None)
        self.assertEqual(provider.window_ms, 1000)
        self.assertEqual(provider.hop_ms, 200)
        self.assertIsNone(provider.veto_window_ms)

    def test_scores_are_raw_cosines_not_clamped_to_unit(self) -> None:
        """Negative raw cosine values pass through unclipped, like offline."""
        scores, ref, _, _ = self._assert_equivalence(600, 100, 600)
        self.assertLess(np.min(scores), -0.5)  # medium zone cosine == -0.8
        self.assertGreater(np.max(scores), 0.5)  # quiet zone cosine == 0.85
        self.assertEqual(np.min(scores), np.min(ref))
        self.assertEqual(np.max(scores), np.max(ref))

    def test_silence_and_embed_none_zones_hold_previous_score(self) -> None:
        """Silent or failing windows reuse the previous score; all finite."""
        scores, _, _, _ = self._assert_equivalence(600, 100, 600)
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertGreater(np.count_nonzero(np.diff(scores) != 0.0), 0)
        # The silence zone must actually hold: >= 0.9 s of identical scores.
        runs = np.diff(np.flatnonzero(np.r_[True, np.diff(scores) != 0.0, True]))
        self.assertGreaterEqual(np.max(runs), 90)

    def test_defaults(self) -> None:
        enrollment = _unit_vector()
        provider = SlidingIdentityProvider(enrollment, _FakeEmbedder(enrollment))
        self.assertEqual(provider.window_ms, 600)
        self.assertEqual(provider.hop_ms, 100)
        self.assertEqual(provider.veto_window_ms, 600)
        self.assertEqual(provider.silence_floor_dbfs, -55.0)
        self.assertEqual(provider.frame_samples, 160)
        self.assertEqual(provider.sample_rate, 16000)
        self.assertEqual(provider.algorithmic_latency_ms, 700.0)
        self.assertFalse(provider.initialized)


class TestProviderContract(unittest.TestCase):
    def setUp(self) -> None:
        self.enrollment = _unit_vector()
        self.fake = _FakeEmbedder(self.enrollment)

    def _provider(self, **kwargs) -> SlidingIdentityProvider:
        return SlidingIdentityProvider(self.enrollment, self.fake, **kwargs)

    def test_observe_before_initialize_raises(self) -> None:
        chunk = np.zeros(FRAME, dtype=np.float32)
        with self.assertRaises(IdentityStreamError):
            self._provider().observe(chunk, sample_rate=SAMPLE_RATE)

    def test_reset_before_initialize_raises(self) -> None:
        with self.assertRaises(IdentityStreamError):
            self._provider().reset()

    def test_double_initialize_raises(self) -> None:
        provider = self._provider()
        provider.initialize()
        with self.assertRaises(IdentityStreamError):
            provider.initialize()

    def test_initialize_requires_embedder(self) -> None:
        with self.assertRaises(IdentityStreamError):
            SlidingIdentityProvider(self.enrollment, None).initialize()

    def test_constructor_rejects_embedder_without_embed(self) -> None:
        with self.assertRaises(IdentityStreamError):
            SlidingIdentityProvider(self.enrollment, object())

    def test_initialize_rejects_zero_norm_enrollment(self) -> None:
        provider = SlidingIdentityProvider(np.zeros(192, dtype=np.float32), self.fake)
        with self.assertRaises(IdentityStreamError):
            provider.initialize()

    def test_observe_rejects_wrong_length_and_rate(self) -> None:
        provider = self._provider()
        provider.initialize()
        with self.assertRaises(IdentityStreamError):
            provider.observe(np.zeros(320, dtype=np.float32), sample_rate=SAMPLE_RATE)
        with self.assertRaises(IdentityStreamError):
            provider.observe(np.zeros(FRAME, dtype=np.float32), sample_rate=8000)

    def test_observe_returns_observation_objects(self) -> None:
        provider = self._provider()
        provider.initialize()
        observation = provider.observe(_square(0.5, FRAME), sample_rate=SAMPLE_RATE)
        self.assertIsInstance(observation, IdentityObservation)
        self.assertEqual(observation.score, 0.0)  # no complete window yet
        self.assertIsNotNone(observation.mixture_energy_db)
        self.assertIsInstance(observation.veto, float)
        self.assertEqual(observation.veto, 0.0)

    def test_validation_rejects_bad_parameters(self) -> None:
        with self.assertRaises(IdentityStreamError):
            self._provider(window_ms=150)
        with self.assertRaises(IdentityStreamError):
            self._provider(window_ms=5000)
        with self.assertRaises(IdentityStreamError):
            self._provider(window_ms=605)  # not frame-aligned
        with self.assertRaises(IdentityStreamError):
            self._provider(hop_ms=5)
        with self.assertRaises(IdentityStreamError):
            self._provider(window_ms=400, hop_ms=500)
        with self.assertRaises(IdentityStreamError):
            self._provider(hop_ms=15)  # not frame-aligned
        with self.assertRaises(IdentityStreamError):
            self._provider(veto_window_ms=605)
        with self.assertRaises(IdentityStreamError):
            self._provider(veto_window_ms=150)
        with self.assertRaises(IdentityStreamError):
            self._provider(window_ms=400, hop_ms=200, veto_window_ms=100)
        with self.assertRaises(IdentityStreamError):
            self._provider(sample_rate=8000)
        with self.assertRaises(IdentityStreamError):
            self._provider(frame_samples=320)
        with self.assertRaises(IdentityStreamError):
            self._provider(silence_floor_dbfs=float("nan"))


class _FakeValue:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    """Minimal session twin: fixed output vector, optionally broken."""

    def __init__(self, output: np.ndarray, inputs: int = 1, outputs: int = 1) -> None:
        self._output = np.asarray(output, dtype=np.float32)
        self._inputs = inputs
        self._outputs = outputs
        self.run_calls = 0

    def get_inputs(self):
        return [_FakeValue("input") for _ in range(self._inputs)]

    def get_outputs(self):
        return [_FakeValue(f"output{i}") for i in range(self._outputs)]

    def run(self, output_names, input_feed):
        self.run_calls += 1
        batch = list(input_feed.values())[0].shape[0]
        return [
            np.broadcast_to(self._output[np.newaxis, :], (batch, self._output.shape[0]))
            for _ in output_names
        ]


class _TrackingFactory:
    def __init__(self, output: np.ndarray, **session_kwargs) -> None:
        self.output = output
        self.session_kwargs = session_kwargs
        self.calls: list[tuple[Path, int]] = []
        self.last_session: _FakeSession | None = None

    def __call__(self, model_path: Path, num_threads: int) -> _FakeSession:
        self.calls.append((Path(model_path), num_threads))
        self.last_session = _FakeSession(self.output, **self.session_kwargs)
        return self.last_session


def _sine_window(seconds: float) -> np.ndarray:
    samples = int(seconds * SAMPLE_RATE)
    return (0.1 * np.sin(2.0 * np.pi * 440.0 * np.arange(samples) / SAMPLE_RATE)).astype(np.float32)


class TestWindowEmbeddingExtractor(unittest.TestCase):
    def _encoder(
        self, output: np.ndarray, num_threads: int = 1, **session_kwargs
    ) -> tuple[WindowEmbeddingExtractor, _TrackingFactory]:
        factory = _TrackingFactory(output, **session_kwargs)
        encoder = WindowEmbeddingExtractor(
            Path("/fake/model.onnx"), num_threads=num_threads, session_factory=factory
        )
        return encoder, factory

    def test_initialize_creates_session_once_and_parses_io(self) -> None:
        vector = _unit_vector(192, seed=9)
        encoder, factory = self._encoder(vector, num_threads=3)
        self.assertFalse(encoder.initialized)
        encoder.initialize()
        self.assertTrue(encoder.initialized)
        self.assertEqual(len(factory.calls), 1)
        self.assertEqual(factory.calls[0][0], Path("/fake/model.onnx").resolve())
        self.assertEqual(factory.calls[0][1], 3)
        encoder.initialize()  # idempotent: no second session
        self.assertEqual(len(factory.calls), 1)

    def test_embed_requires_initialize(self) -> None:
        encoder, _ = self._encoder(_unit_vector(192, seed=9))
        with self.assertRaises(IdentityStreamError):
            encoder.embed(_sine_window(0.6))

    def test_embed_never_creates_sessions(self) -> None:
        vector = _unit_vector(192, seed=9)
        encoder, factory = self._encoder(vector)
        encoder.initialize()
        for _ in range(5):
            self.assertIsNotNone(encoder.embed(_sine_window(0.6)))
        self.assertEqual(len(factory.calls), 1)  # embed() only reuses the session

    def test_embed_normalizes_fixed_output(self) -> None:
        vector = _unit_vector(192, seed=9)
        encoder, _ = self._encoder(vector)
        encoder.initialize()
        result = encoder.embed(_sine_window(0.6))
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (192,))
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-6)
        np.testing.assert_allclose(result, vector, atol=1e-6)

    def test_embed_accepts_window_shorter_than_3s(self) -> None:
        vector = _unit_vector(192, seed=9)
        encoder, _ = self._encoder(vector)
        encoder.initialize()
        result = encoder.embed(_sine_window(0.6))  # 600 ms window
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (192,))

    def test_embed_too_short_returns_none_without_session_call(self) -> None:
        encoder, factory = self._encoder(_unit_vector(192, seed=9))
        encoder.initialize()
        self.assertIsNone(encoder.embed(np.zeros(100, dtype=np.float32)))
        self.assertIsNotNone(factory.last_session)
        self.assertEqual(factory.last_session.run_calls, 0)

    def test_embed_zero_norm_output_returns_none(self) -> None:
        encoder, factory = self._encoder(np.zeros(192, dtype=np.float32))
        encoder.initialize()
        self.assertIsNone(encoder.embed(_sine_window(0.6)))
        self.assertIsNotNone(factory.last_session)
        self.assertEqual(factory.last_session.run_calls, 1)  # ran, then rejected

    def test_initialize_rejects_multi_output_model(self) -> None:
        encoder, _ = self._encoder(_unit_vector(192, seed=9), outputs=2)
        with self.assertRaises(IdentityStreamError):
            encoder.initialize()

    def test_default_factory_checks_model_file(self) -> None:
        encoder = WindowEmbeddingExtractor(Path("/nonexistent/model.onnx"), num_threads=1)
        with self.assertRaises(IdentityStreamError):
            encoder.initialize()

    def test_embed_enrollment_requires_3s(self) -> None:
        vector = _unit_vector(192, seed=9)
        encoder, _ = self._encoder(vector)
        encoder.initialize()
        with self.assertRaises(IdentityStreamError):
            encoder.embed_enrollment(_sine_window(2.0))
        result = encoder.embed_enrollment(_sine_window(3.0))
        self.assertEqual(result.shape, (192,))
