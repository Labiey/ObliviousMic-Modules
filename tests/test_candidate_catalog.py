import io
import json
from types import SimpleNamespace

import pytest
from obliviousmic.catalog import OfficialCatalog
from obliviousmic.plugin_manager import PluginManager
from obliviousmic.sdk import ModuleRegistry

from tools import publish_release
from tools.prepare_catalog import ROOT, prepare
from tools.validate_catalog import validate


def test_candidate_is_deterministic_accepted_by_host_and_preserves_published_entries(tmp_path):
    published = (ROOT / "catalog/index.json").read_bytes()
    first = prepare(tmp_path / "a" / "catalog.next.json", include_models=False)
    second = prepare(tmp_path / "b" / "catalog.next.json", include_models=False)
    assert first.read_bytes() == second.read_bytes()
    assert (ROOT / "catalog/index.json").read_bytes() == published
    candidate = json.loads(first.read_text("utf-8"))
    assert {x["name"] for x in candidate["modules"]} == {
        "campplus",
        "bsrnn",
        "asr",
        "asr_api",
        "tts",
        "subtitles",
        "broadcast",
        "translate",
        "denoise",
    }
    validate(first)
    catalog = OfficialCatalog(
        tmp_path, None, None, opener=lambda *_a, **_k: io.BytesIO(first.read_bytes())
    )
    assert all(item["compatible"] for item in catalog.read()["modules"])
    duplicate = {**candidate, "modules": candidate["modules"] + candidate["modules"][:1]}
    first.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(AssertionError, match="duplicate"):
        validate(first)


def test_candidate_installs_four_independent_packages_through_catalog_and_restores(tmp_path):
    path = prepare(tmp_path / "packages" / "catalog.next.json", include_models=False)
    candidate = json.loads(path.read_text("utf-8"))
    prepared = []
    registry = ModuleRegistry()
    manager = PluginManager(tmp_path / "data", registry)

    def open(request, **_kwargs):
        url = request.full_url
        if url.endswith("catalog/index.json"):
            return io.BytesIO(path.read_bytes())
        item = next(x for x in candidate["modules"] if x["download_url"] == url)
        response = io.BytesIO((path.parent / url.rsplit("/", 1)[1]).read_bytes())
        response.geturl = lambda: item["download_url"]
        return response

    setup = SimpleNamespace(prepare=lambda module_id, job: prepared.append(module_id))
    job = SimpleNamespace(progress=lambda *_: None, checkpoint=lambda: None, commit=lambda fn: fn())
    catalog = OfficialCatalog(tmp_path / "data", manager, setup, opener=open)
    for item in candidate["modules"][2:]:
        record = catalog.install(item["module_id"], job, item["sha256"])
        assert record["bundle_sha256"] == item["sha256"]
    assert set(prepared) == {
        "org.obliviousmic." + name
        for name in ("asr", "asr_api", "tts", "subtitles", "broadcast", "translate", "denoise")
    }
    restored = PluginManager(tmp_path / "data", ModuleRegistry())
    assert restored.restore() == []
    assert {x["module_id"] for x in restored.discover()} == set(prepared)


def test_release_preflight_checks_supported_tag_and_catalog_digest_before_network(
    tmp_path, monkeypatch
):
    candidate = prepare(tmp_path / "catalog.next.json", include_models=False)
    monkeypatch.setattr(publish_release, "prepare", lambda **_kwargs: candidate)
    monkeypatch.setattr(
        publish_release, "request", lambda *_a, **_k: pytest.fail("unexpected publish")
    )
    current = json.loads((ROOT / "modules/tts/obliviousmic.module.json").read_text("utf-8"))[
        "module_version"
    ]
    name, version, package = publish_release.audited_package("tts-v" + current)
    assert (name, version, package["module_id"]) == ("tts", current, "org.obliviousmic.tts")
    for tag in ("tts-v99.0.0", "unlisted-v1.0.0", "../tts-v1.0.0"):
        with pytest.raises(ValueError, match="not in"):
            publish_release.audited_package(tag)
    data = json.loads(candidate.read_text("utf-8"))
    next(x for x in data["modules"] if x["name"] == "tts")["sha256"] = "0" * 64
    candidate.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        publish_release.audited_package("tts-v" + current)
