import json
from pathlib import Path

from obliviousmic.package_declaration import read_declaration

ROOT = Path(__file__).resolve().parents[1]


def test_all_maintained_sources_use_complete_unified_declarations_and_matching_host_schemas():
    from obliviousmic.package_declaration import declaration_schema

    for kind in ("module", "theme"):
        assert json.loads(
            (ROOT / f"schemas/obliviousmic-{kind}-v1.json").read_text("utf-8")
        ) == declaration_schema(kind)
    count = 0
    for folder in sorted((ROOT / "modules").iterdir()):
        if not folder.is_dir() or folder.name.startswith("_"):
            continue
        document = read_declaration(folder, "module")
        assert document["module_id"] == "org.obliviousmic." + folder.name
        assert set(document["display_names"]) == {"zh-CN", "en", "ja"}
        assert all(document["display_names"].values()) and document["description"]
        assert document["repository"] == "https://github.com/Labiey/ObliviousMic-Modules"
        assert not (folder / "manifest.json").exists()
        count += 1
    assert count == 9
