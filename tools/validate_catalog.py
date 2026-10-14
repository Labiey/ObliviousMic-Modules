"""Validate published declarations and deterministic package digests without executing code."""

import ast
import json
from pathlib import Path

from tools.build_module import build

ROOT = Path(__file__).resolve().parents[1]


def validate():
    catalog = json.loads((ROOT / "catalog/index.json").read_text("utf-8"))
    assert (
        catalog["catalog_version"] == 1 and catalog["repository"] == "Labiey/ObliviousMic-Modules"
    )
    for item in catalog["modules"]:
        name = item["name"]
        folder = ROOT / "modules" / name
        package = build(name)
        assert item["sha256"] == package["sha256"] and item["bytes"] == package["bytes"]
        assert item["module_id"] == package["module_id"] and item["version"] == package["version"]
        assert (folder / "LICENSE").is_file() and (folder / "NOTICE").is_file()
        for path in folder.rglob("*.py"):
            ast.parse(path.read_text("utf-8-sig"))
        resources = json.loads((folder / "resources.json").read_text("utf-8"))["assets"]
        for resource in resources:
            if resource["auto_download"]:
                assert (
                    resource["redistribution"] == "allowed" and resource["license"] == "Apache-2.0"
                )
                assert len(resource["sha256"]) == 64 and resource["bytes"] > 0
            if resource["redistribution"] != "allowed":
                assert not resource["auto_download"]
    print("PASS: catalog digests, declarations, code-only packages and licensing metadata")


if __name__ == "__main__":
    validate()
