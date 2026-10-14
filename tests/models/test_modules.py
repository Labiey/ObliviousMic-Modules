import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from obliviousmic.__main__ import builtin_registry
from obliviousmic.adapters import PythonAdapter
from obliviousmic.graph.dag import DAGCompiler
from obliviousmic.graph.dag_runtime import DAGRuntime
from obliviousmic.sdk import (
    AudioFormat,
    AudioPacket,
    ModuleRegistry,
    PortBundle,
    PrepareContext,
    StreamContext,
)

from modules.bsrnn.lib.frame_glue import FrameBackend, periodic_hann
from modules.campplus.lib.identity_stream import SlidingIdentityProvider
from modules.registration import register_models


class Embedder:
    def embed(self, x):
        return np.array([1.0, 0.0], np.float32)


class IdentityBackend(FrameBackend):
    def initial_states(self):
        return ()

    def separate_frames(self, frames, layers):
        return np.fft.rfft(frames * periodic_hann(512), axis=-1), layers


def identity_factory(context, config):
    return SlidingIdentityProvider(
        np.array([1.0, 0.0], np.float32),
        Embedder(),
        window_ms=config["window_ms"],
        hop_ms=config["hop_ms"],
        veto_window_ms=config["veto_window_ms"],
    )


def test_camplus_matches_leaf_frame_scores_and_preserves_observed_and_effective_time():
    registry = ModuleRegistry()
    register_models(registry, identity_factory=identity_factory)
    registration = registry.get("org.obliviousmic.campplus")
    adapter = PythonAdapter(registration, "identity")
    adapter.prepare(PrepareContext("identity", {"audio": AudioFormat(16000)}, {}), {})
    adapter.start(StreamContext())
    reference = identity_factory(None, registration.settings({}))
    reference.initialize()
    x = np.random.default_rng(10).uniform(-0.1, 0.1, 16000).astype(np.float32)
    for start in range(0, len(x), 160):
        p = AudioPacket(x[start : start + 160, None], AudioFormat(16000), Fraction(start, 16000))
        metadata = adapter.process(PortBundle.audio(p)).ports["decision"][0]
        expected = reference.observe(p.samples[:, 0], sample_rate=16000)
        assert metadata.value["score"] == expected.score
        assert metadata.value["veto"] == expected.veto
        assert metadata.value["effective_start"] == [
            p.media_start.numerator,
            p.media_start.denominator,
        ]
        assert metadata.observed_end <= p.media_end
        assert metadata.private
    assert adapter.flush().drained
    adapter.dispose()


def test_private_parallel_graph_has_no_unverified_raw_fallback_and_keeps_eos_length():
    registry = builtin_registry()
    # Replace explicit fixtures in a fresh registry, keeping trusted core declarations.
    fixtures = ModuleRegistry()
    for module in registry.describe():
        if module["module_id"] not in {"org.obliviousmic.campplus", "org.obliviousmic.bsrnn"}:
            fixtures.register(registry.get(module["module_id"]))
    register_models(
        fixtures, identity_factory=identity_factory, separation_factory=lambda *_: IdentityBackend()
    )
    g = json.loads((Path(__file__).resolve().parents[2] / "examples/private_dag.json").read_text())
    plan = DAGCompiler(fixtures).compile(g, AudioFormat(16000))
    x = np.full((16001, 1), 0.1, np.float32)
    output = []
    with DAGRuntime(plan) as runtime:
        for start in range(0, len(x), 333):
            output.extend(
                runtime.process(
                    AudioPacket(x[start : start + 333], AudioFormat(16000), Fraction(start, 16000))
                )
            )
        output.extend(runtime.flush())
    samples = np.concatenate([p.samples for p in output])
    assert len(samples) == len(x)
    np.testing.assert_array_equal(samples[: 160 * 59], 0)
    assert np.any(samples[160 * 60 :] != 0)
