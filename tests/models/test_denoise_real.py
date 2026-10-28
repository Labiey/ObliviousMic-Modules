"""Opt-in real weights: set OBMIC_DPDFNET_TEST_CACHE to a verified build cache."""

import json
import os
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from obliviousmic.__main__ import builtin_registry
from obliviousmic.graph.dag import DAGCompiler
from obliviousmic.graph.dag_runtime import DAGRuntime
from obliviousmic.module_setup import ModuleSetup
from obliviousmic.plugin_manager import PluginManager
from obliviousmic.profiles import AssetStore, ProfileStore
from obliviousmic.resources import ResourcePool
from obliviousmic.sdk import AudioFormat, AudioPacket

from tools.build_module import ROOT, build


@pytest.mark.parametrize("mode", ["light", "enhanced"])
def test_full_installer_binds_models_offline_and_runs_format_negotiated_graph(tmp_path, mode):
    cache = os.environ.get("OBMIC_DPDFNET_TEST_CACHE")
    if not cache:
        pytest.skip("opt-in verified DPDFNet model cache is not configured")
    package = build(
        "denoise", tmp_path / "packages", include_models=True, cache=Path(cache), download=False
    )
    registry = builtin_registry()
    manager = PluginManager(tmp_path / "data", registry)
    record = manager.install(package["bundle_path"], trusted=True, official=True)
    items = json.loads((ROOT / "modules/denoise/resources.json").read_text("utf-8"))["assets"]
    assert set(record["bundled_assets"]) == {r["sha256"] for r in items}
    assets = AssetStore(manager.root)
    setup = ModuleSetup(
        manager.root,
        registry,
        assets,
        ProfileStore(manager.root),
        opener=lambda *_a, **_k: pytest.fail("a complete installer must not download weights"),
    )
    job = SimpleNamespace(checkpoint=lambda: None, progress=lambda *_: None, commit=lambda fn: fn())
    setup.prepare("org.obliviousmic.denoise", job)
    registration = registry.get("org.obliviousmic.denoise")
    values = setup.fill(registration, registration.settings({"mode": mode}), None)
    assert setup.status("org.obliviousmic.denoise", settings=values)["state"] == "ready"
    assert values["light_model_asset"] == items[0]["sha256"]
    assert values["enhanced_model_asset"] == items[1]["sha256"]
    graph = {
        "graph_schema": 1,
        "graph_id": "denoise-real",
        "revision": 1,
        "execution": "dag",
        "nodes": [
            {"id": "source", "type": "core.source"},
            {"id": "denoise", "type": "org.obliviousmic.denoise", "settings": values},
            {"id": "sink", "type": "core.sink"},
        ],
        "edges": [
            {"from": "source.audio", "to": "denoise.audio"},
            {"from": "denoise.audio", "to": "sink.audio"},
        ],
    }
    pool = ResourcePool()
    samples = np.random.default_rng(9).normal(0, 0.02, (16000 * 3 + 17, 2)).astype(np.float32)
    fmt = AudioFormat(16000, 2)
    compiled = DAGCompiler(registry).compile(graph, fmt, desired_output_format=fmt)
    with DAGRuntime(
        compiled,
        capabilities=lambda _: {
            "asset.read": assets.resolve,
            "resource.acquire": pool.acquire_session,
        },
    ) as runtime:
        outputs = []
        for start in range(0, len(samples), 160):
            outputs.extend(
                runtime.process(
                    AudioPacket(samples[start : start + 160], fmt, Fraction(start, 16000))
                )
            )
        outputs.extend(runtime.flush())
        result = np.concatenate([p.samples for p in outputs])
        assert result.shape == samples.shape and np.isfinite(result).all()
        assert outputs[0].media_start == 0 and outputs[-1].media_end == Fraction(
            len(samples), 16000
        )
        assert np.sqrt(np.mean(result[16000:] ** 2)) < np.sqrt(np.mean(samples[16000:] ** 2)) * 0.5
        assert runtime.adapters["denoise"].report_monitor()["enhanced"] == (mode == "enhanced")
    assert all(entry["refs"] == 0 for entry in pool.entries.values())
