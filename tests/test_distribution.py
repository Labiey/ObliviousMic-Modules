import zipfile

import pytest
from obliviousmic.plugin_manager import PluginManager
from obliviousmic.sdk import ModuleRegistry

from tools.build_module import build


@pytest.mark.parametrize("name", ["campplus", "bsrnn", "asr", "tts", "subtitles", "broadcast"])
def test_independent_bundle_roundtrip_and_licenses(tmp_path, name):
    one = build(name, tmp_path / "a")
    two = build(name, tmp_path / "b")
    assert one["sha256"] == two["sha256"]
    with zipfile.ZipFile(one["bundle_path"]) as archive:
        names = archive.namelist()
        assert names == sorted(names)
        assert {"LICENSE", "NOTICE", "README.md", "resources.json"} <= set(names)
        if name in {"campplus", "bsrnn"}:
            assert {"licensing.md", "provenance.json"} <= set(names)
        assert not any(x.endswith((".onnx", ".wav", ".npy")) for x in names)
    manager = PluginManager(tmp_path / "data", ModuleRegistry())
    with pytest.raises(ValueError, match="reserved"):
        manager.install(one["bundle_path"], trusted=True)
    manager.install(one["bundle_path"], trusted=True, official=True)
    registration = manager.registry.get("org.obliviousmic." + name)
    if name in {"campplus", "bsrnn"}:
        assert registration.setup["initialize_action"] == "initialize"
    elif name == "tts":
        assert registration.setup["initialize_action"] == "initialize_reference"
    recovered = PluginManager(tmp_path / "data", ModuleRegistry())
    assert recovered.restore() == []
    assert (
        recovered.registry.get(registration.manifest.module_id).resources == registration.resources
    )
    if name == "bsrnn":
        template = next(x for x in registration.resources if x["id"] == "template")
        assert not template["auto_download"] and template["redistribution"] == "unresolved"
