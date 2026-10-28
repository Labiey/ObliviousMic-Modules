from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
from obliviousmic.resources import ResourcePool, reservation_bytes, session_key
from obliviousmic.sdk import AudioFormat, AudioPacket, PortBundle, PrepareContext, StreamContext

from modules.denoise.engine import HOP, LATENCY_FRAMES, RATE, Stream
from modules.denoise.runtime import Module


class DelayedIdentity:
    """Model-shaped spectral delay; verifies compensation independently of weights."""

    initial_state = np.zeros(4 * 481 * 2, np.float32)

    def infer(self, spectrum, state):
        queue = state.reshape(4, 1, 1, 481, 2)
        return queue[0].copy(), np.concatenate((state[481 * 2 :], spectrum.ravel()))


def process_all(engine, samples, chunks, mix=1):
    result, position, i = [], 0, 0
    while position < len(samples):
        count = chunks[i % len(chunks)]
        result.append(engine.process(samples[position : position + count], mix))
        position += count
        i += 1
    result.append(engine.flush(mix))
    assert not engine.flush(mix).size
    return np.concatenate(result)


@pytest.mark.parametrize("length", [1, 479, 480, 481, 2400, 2401, 7777])
@pytest.mark.parametrize("mix", [0, 0.35, 1])
def test_exact_eos_and_stft_identity_including_first_and_last_sample(length, mix):
    samples = np.random.default_rng(34).uniform(-0.4, 0.4, length).astype(np.float32)
    out = process_all(Stream(DelayedIdentity()), samples, [1, 107, 480, 47], mix)
    assert len(out) == length
    np.testing.assert_allclose(out, samples, atol=1e-7)


def test_latency_independent_state_reset_and_bounded_input_ownership():
    shared = DelayedIdentity()
    a, b = Stream(shared), Stream(shared)
    scratch = np.full(HOP, 0.2, np.float32)
    for i in range(100):
        out = a.process(scratch)
        silent = b.process(np.zeros(HOP, np.float32))
        if i < LATENCY_FRAMES // HOP:
            assert not out.size
        else:
            np.testing.assert_allclose(out, 0.2, atol=1e-7)
        assert not np.any(silent)
        assert len(a.pending) == HOP
        assert sum(map(len, a.dry)) <= LATENCY_FRAMES
    scratch[:] = 1  # Capture buffers may be reused after the call returns.
    np.testing.assert_allclose(a.flush(), 0.2, atol=1e-7)
    a.reset()
    assert not np.any(process_all(a, np.zeros(2001, np.float32), [480]))
    with pytest.raises(RuntimeError, match="drained"):
        a.process(scratch)


def config(**changes):
    return {**{k: v["default"] for k, v in Module.SCHEMA["properties"].items()}, **changes}


def test_module_pts_discontinuity_hot_parameters_and_tail():
    module = Module(lambda *_: DelayedIdentity())
    module.prepare(
        PrepareContext("dn", {"audio": AudioFormat(RATE)}, {"audio": AudioFormat(RATE)}), config()
    )
    module.start(StreamContext())
    out = []
    for start in range(0, 6001, 480):
        packet = AudioPacket(
            np.full((min(480, 6001 - start), 1), 0.3, np.float32),
            AudioFormat(RATE),
            Fraction(2) + Fraction(start, RATE),
        )
        out.extend(module.process(PortBundle.audio(packet)).ports["audio"])
    out.extend(module.flush().outputs.ports["audio"])
    assert out[0].media_start == 2 and out[-1].media_end == 2 + Fraction(6001, RATE)
    assert all(a.media_end == b.media_start for a, b in zip(out, out[1:], strict=False))
    assert sum(p.frames for p in out) == 6001
    assert module.report_latency().seconds == Fraction(1, 20)
    module.update_parameters({"mix": 0.2})
    with pytest.raises(ValueError, match="reprepare"):
        module.update_parameters({"mode": "enhanced"})
    for value in [True, float("nan"), 2]:
        with pytest.raises(ValueError, match="mix"):
            module.update_parameters({"mix": value})
    packet = AudioPacket(np.zeros((17, 1), np.float32), AudioFormat(RATE), Fraction(20), epoch_id=1)
    assert not module.process(PortBundle.audio(packet)).ports["audio"]
    tail = module.flush().outputs.ports["audio"]
    assert tail[0].epoch_id == 1 and tail[0].media_start == 20 and tail[0].frames == 17
    assert not np.any(tail[0].samples)
    module.dispose()


def test_managed_model_selection_pool_keys_and_release(tmp_path, monkeypatch):
    from modules.denoise import runtime

    calls = []
    monkeypatch.setattr(
        runtime,
        "Session",
        lambda path, mode, threads: calls.append((mode, threads)) or DelayedIdentity(),
    )
    pool = ResourcePool(1024 * 1024)
    paths = {mode: tmp_path / mode for mode in ("light", "enhanced")}
    for mode, path in paths.items():
        path.write_bytes(mode.encode())

    def acquire(path, settings, factory):
        return pool.acquire(session_key(path, settings), reservation_bytes(path, settings), factory)

    context = PrepareContext(
        "dn",
        {"audio": AudioFormat(RATE)},
        {"audio": AudioFormat(RATE)},
        capabilities={"asset.read": lambda key: paths[key], "resource.acquire": acquire},
    )
    modules = [Module(), Module(), Module()]
    for instance, mode in zip(modules, ("light", "light", "enhanced"), strict=True):
        instance.prepare(
            context, config(mode=mode, light_model_asset="light", enhanced_model_asset="enhanced")
        )
    assert calls == [("light", 1), ("enhanced", 1)]
    assert modules[0].processor.state is not modules[1].processor.state
    for instance in modules:
        instance.dispose()
    assert all(entry["refs"] == 0 for entry in pool.entries.values())


def test_failed_stream_initialization_releases_session():
    module = Module()
    released = []
    lease = SimpleNamespace(value=object(), release=lambda: released.append(True))
    context = PrepareContext(
        "dn",
        {},
        {},
        capabilities={"asset.read": lambda _: "unused", "resource.acquire": lambda *_: lease},
    )
    with pytest.raises(AttributeError):
        module.prepare(context, config())
    assert released == [True] and module.lease is None


def test_async_graph_merge_preserves_media_alignment_and_exact_tail():
    from obliviousmic.__main__ import builtin_registry
    from obliviousmic.graph.dag import DAGCompiler
    from obliviousmic.graph.dag_runtime import DAGRuntime
    from obliviousmic.sdk import Manifest, ModuleRegistration

    registry = builtin_registry()
    registry.register(
        ModuleRegistration(
            Manifest(Module.MANIFEST), Module.SCHEMA, lambda: Module(lambda *_: DelayedIdentity())
        )
    )
    graph = {
        "graph_schema": 1,
        "graph_id": "denoise",
        "revision": 1,
        "execution": "dag",
        "nodes": [
            {"id": "source", "type": "core.source"},
            {"id": "dn", "type": "org.obliviousmic.denoise"},
            {"id": "merge", "type": "core.merge"},
            {"id": "sink", "type": "core.sink"},
        ],
        "edges": [
            {"from": a, "to": b}
            for a, b in [
                ("source.audio", "dn.audio"),
                ("dn.audio", "merge.in1"),
                ("source.audio", "merge.in2"),
                ("merge.audio", "sink.audio"),
            ]
        ],
    }
    samples = np.random.default_rng(3).uniform(-0.2, 0.2, (9601, 1)).astype(np.float32)
    with DAGRuntime(DAGCompiler(registry).compile(graph, AudioFormat(RATE))) as runtime:
        outputs = []
        for start in range(0, len(samples), 480):
            packet = AudioPacket(
                samples[start : start + 480], AudioFormat(RATE), Fraction(start, RATE)
            )
            outputs.extend(runtime.process(packet))
        outputs.extend(runtime.flush())
        result = np.concatenate([p.samples for p in outputs])
        assert outputs[0].media_start == 0 and outputs[-1].media_end == Fraction(len(samples), RATE)
        np.testing.assert_allclose(result, samples, atol=1e-7)
