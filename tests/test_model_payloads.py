import hashlib
import json
import zipfile

import pytest
from obliviousmic.sdk.declaration import audio, manifest

from tools import build_module
from tools.model_payloads import acquire


def environment(tmp_path, monkeypatch, **change):
    folder = tmp_path / "source/modules/fixture"
    (folder / "licenses/model").mkdir(parents=True)
    data = b"synthetic redistributable model"
    resource = {
        "id": "weights",
        "module_id": "example.fixture",
        "format": "binary",
        "license": "MIT",
        "license_file": "licenses/model/LICENSE",
        "redistribution": "allowed",
        "auto_download": True,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "source_url": "https://example.com/model",
        "download_url": "https://example.com/model.bin",
        **change,
    }
    (folder / "obliviousmic.module.json").write_text(
        json.dumps(manifest("example.fixture", [audio()], [audio()]))
    )
    (folder / "resources.json").write_text(
        json.dumps({"resources_version": 1, "assets": [resource]})
    )
    (folder / "licenses/model/LICENSE").write_text("Synthetic test license fixture")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / resource["sha256"]).write_bytes(data)
    monkeypatch.setattr(build_module, "ROOT", tmp_path / "source")
    return resource, cache


def test_release_payload_is_deterministic_and_code_only_is_separate(tmp_path, monkeypatch):
    item, cache = environment(tmp_path, monkeypatch)
    one = build_module.build(
        "fixture", tmp_path / "a", include_models=True, cache=cache, download=False
    )
    two = build_module.build(
        "fixture", tmp_path / "b", include_models=True, cache=cache, download=False
    )
    assert one["sha256"] == two["sha256"]
    with zipfile.ZipFile(one["bundle_path"]) as archive:
        assert archive.namelist() == sorted(archive.namelist())
        resources = json.loads(archive.read("resources.json"))["assets"]
        assert resources[0]["bundled_path"] == "models/weights.bin"
        assert hashlib.sha256(archive.read("models/weights.bin")).hexdigest() == item["sha256"]
        assert "licenses/model/LICENSE" in archive.namelist()
    code = build_module.build("fixture", tmp_path / "code")
    assert code["bundled_resources"] == []
    with zipfile.ZipFile(code["bundle_path"]) as archive:
        assert not any(name.startswith("models/") for name in archive.namelist())


def test_unresolved_weights_stay_out_and_failure_does_not_leave_a_code_installer(
    tmp_path, monkeypatch
):
    _, cache = environment(tmp_path, monkeypatch, redistribution="unresolved", auto_download=False)
    package = build_module.build("fixture", tmp_path / "out", include_models=True, cache=cache)
    assert package["bundled_resources"] == []
    with zipfile.ZipFile(package["bundle_path"]) as archive:
        assert not any(name.startswith("models/") for name in archive.namelist())
    folder = tmp_path / "source/modules/fixture"
    resources = json.loads((folder / "resources.json").read_text())
    resources["assets"][0]["auto_download"] = True
    (folder / "resources.json").write_text(json.dumps(resources))
    with pytest.raises(ValueError, match="unverified"):
        build_module.build("fixture", tmp_path / "failed", include_models=True, cache=cache)
    assert not list((tmp_path / "failed").glob("*.zip"))


def test_cache_integrity_missing_cache_and_download_transport(tmp_path, monkeypatch):
    item, cache = environment(tmp_path, monkeypatch)
    assert acquire(item, cache, download=False).name == item["sha256"]
    (cache / item["sha256"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        acquire(item, cache)
    with pytest.raises(FileNotFoundError, match="missing"):
        acquire(item, tmp_path / "empty", download=False)
    with pytest.raises(ValueError, match="HTTPS"):
        acquire({**item, "download_url": "http://example.com/model"}, tmp_path / "empty")
