import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
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
from modules.campplus.lib.presence import StreamingPresenceGate
from modules.campplus.runtime import Module as CAM
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


def test_camplus_keeps_leaf_scores_in_monitor_and_outputs_only_boolean_controls():
    registry = ModuleRegistry()
    register_models(registry, identity_factory=identity_factory)
    registration = registry.get("org.obliviousmic.campplus")
    adapter = PythonAdapter(registration, "identity")
    adapter.prepare(PrepareContext("identity", {"audio": AudioFormat(16000)}, {}), {})
    adapter.start(StreamContext())
    reference = identity_factory(None, registration.settings({}))
    reference.initialize()
    policy = StreamingPresenceGate(CAM.policy_config(registration.settings({})), frame_hz=100)
    x = np.random.default_rng(10).uniform(-0.1, 0.1, 16000).astype(np.float32)
    for start in range(0, len(x), 160):
        p = AudioPacket(x[start : start + 160, None], AudioFormat(16000), Fraction(start, 16000))
        decision = adapter.process(PortBundle.audio(p)).ports["decision"][0]
        expected = reference.observe(p.samples[:, 0], sample_rate=16000)
        monitor = adapter.report_monitor()
        assert monitor["score"] == expected.score
        assert type(decision.value) is bool
        assert decision.value == policy.update(
            expected.score, expected.mixture_energy_db, expected.mixture_energy_db, expected.veto
        )
        assert monitor["confidence"] == policy.confidence
        assert decision.effective_from == p.media_start
        assert decision.late_policy == "apply_next"
    assert adapter.flush().drained
    adapter.dispose()


def test_identity_threshold_hot_update_preserves_session_and_rejects_invalid_pair():
    from types import SimpleNamespace

    class Provider:
        def initialize(self):
            pass

        def reset(self):
            pass

        def observe(self, frame, **kwargs):
            return SimpleNamespace(score=0.4, veto=0.4, mixture_energy_db=-30.0)

    calls = []

    def factory(*_):
        calls.append(True)
        return Provider()

    registry = ModuleRegistry()
    register_models(registry, identity_factory=factory)
    reg = registry.get("org.obliviousmic.campplus")
    adapter = PythonAdapter(reg, "cam")
    adapter.prepare(PrepareContext("cam", {"audio": AudioFormat(16000)}, {}), {})
    adapter.start(StreamContext())

    def step(index):
        p = AudioPacket(
            np.full((160, 1), 0.1, np.float32), AudioFormat(16000), Fraction(index, 100)
        )
        return adapter.process(PortBundle.audio(p)).ports["decision"][0].value

    assert not any(step(i) for i in range(30))
    with pytest.raises(ValueError, match="close_threshold"):
        adapter.update_parameters({"open_threshold": 0.2}, expected_revision=0)
    adapter.update_parameters({"open_threshold": 0.33, "close_threshold": 0.3}, expected_revision=0)
    states = [step(i) for i in range(30, 60)]
    assert any(states) and len(calls) == 1
    assert adapter.parameter_revision == 1
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


@pytest.mark.parametrize("delay_ms,retained", [(0, 0), (400, 1600)])
def test_onset_delay_before_boolean_gate_retains_the_initial_speech(delay_ms, retained):
    from types import SimpleNamespace

    class Provider:
        def initialize(self):
            pass

        def reset(self):
            self.index = 0

        def observe(self, frame, **kwargs):
            self.index += 1
            return SimpleNamespace(
                score=0.9 if self.index > 30 else 0.0, veto=None, mixture_energy_db=-20.0
            )

    r = builtin_registry()
    register_models(r, identity_factory=lambda *_: Provider())
    g = {
        "graph_schema": 1,
        "graph_id": "onset",
        "revision": 1,
        "execution": "dag",
        "nodes": [
            {"id": "source", "type": "core.source"},
            {
                "id": "cam",
                "type": "org.obliviousmic.campplus",
                "settings": {"open_hold_ms": 0, "min_close_ms": 0},
            },
            {"id": "delay", "type": "core.delay", "settings": {"delay_ms": delay_ms}},
            {"id": "gate", "type": "core.gate"},
            {"id": "sink", "type": "core.sink"},
        ],
        "edges": [
            {"from": a, "to": b}
            for a, b in [
                ("source.audio", "cam.audio"),
                ("source.audio", "delay.audio"),
                ("delay.audio", "gate.audio"),
                ("cam.decision", "gate.control"),
                ("gate.audio", "sink.audio"),
            ]
        ],
    }
    output = []
    with DAGRuntime(DAGCompiler(r).compile(g, AudioFormat(16000))) as rt:
        for i in range(100):
            p = AudioPacket(
                np.full((160, 1), 0.1 if i < 10 else 0, np.float32),
                AudioFormat(16000),
                Fraction(i, 100),
            )
            output.extend(rt.process(p))
        output.extend(rt.flush())
    samples = np.concatenate([p.samples for p in output])
    assert np.count_nonzero(samples) == retained
    assert len(samples) == 16000 + delay_ms * 16


def test_async_boolean_analysis_can_change_state_after_a_late_result():
    from types import SimpleNamespace

    class Provider:
        def initialize(self):
            pass

        def reset(self):
            pass

        def observe(self, frame, **kwargs):
            return SimpleNamespace(score=float(frame[0]), veto=None, mixture_energy_db=-20)

    r = builtin_registry()
    register_models(r, identity_factory=lambda *_: Provider())
    g = json.loads((Path(__file__).resolve().parents[2] / "examples/private_dag.json").read_text())
    g["nodes"] = [n for n in g["nodes"] if n["id"] != "separation"]
    g["edges"] = [e for e in g["edges"] if "separation" not in str(e)]
    g["edges"].append({"from": "source.audio", "to": "gate.audio"})
    next(n for n in g["nodes"] if n["id"] == "identity")["settings"] = {
        "open_hold_ms": 0,
        "close_hold_ms": 0,
        "min_close_ms": 0,
        "min_open_ms": 0,
        "ema_open_alpha": 1,
        "ema_close_alpha": 1,
    }
    with DAGRuntime(DAGCompiler(r).compile(g, AudioFormat(16000)), mode="realtime") as rt:
        for i in range(80):
            samples = np.full((160, 1), 0.1 if i < 20 else 0.9, np.float32)
            rt.process(AudioPacket(samples, AudioFormat(16000), Fraction(i, 100)))
            if "identity" in rt.async_jobs:
                rt.async_jobs["identity"][0].result(timeout=2)
            rt.process_endpoints({})
        report = {n["id"]: n for n in rt.monitor_report()["nodes"]}
        assert report["gate"]["gate"]["opened"]
        assert report["identity"]["gate"]["confidence"] == pytest.approx(0.9)
        assert rt.state == "running"
