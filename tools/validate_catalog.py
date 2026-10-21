"""Validate published declarations and deterministic package digests without executing code."""

import argparse
import ast
import json
from pathlib import Path

from tools.build_module import build

ROOT = Path(__file__).resolve().parents[1]


def validate(path=None):
    catalog = json.loads(Path(path or ROOT / "catalog/index.json").read_text("utf-8"))
    assert (
        catalog["catalog_version"] == 1 and catalog["repository"] == "Labiey/ObliviousMic-Modules"
    )
    ids = [item["module_id"] for item in catalog["modules"]]
    assert len(ids) == len(set(ids)), "duplicate catalog module"
    for item in catalog["modules"]:
        name = item["name"]
        folder = ROOT / "modules" / name
        package = build(name)
        assert item["sha256"] == package["sha256"] and item["bytes"] == package["bytes"], (
            f"{name}: catalog {item['sha256']} / {item['bytes']} bytes; "
            f"built {package['sha256']} / {package['bytes']} bytes"
        )
        assert item["module_id"] == package["module_id"] and item["version"] == package["version"]
        assert (folder / "LICENSE").is_file() and (folder / "NOTICE").is_file()
        for path in folder.rglob("*.py"):
            ast.parse(path.read_text("utf-8-sig"))
        resources = json.loads((folder / "resources.json").read_text("utf-8"))["assets"]
        for resource in resources:
            if resource["auto_download"]:
                assert resource["redistribution"] == "allowed" and resource["license"] in {
                    "Apache-2.0",
                    "MIT",
                }
                assert len(resource["sha256"]) == 64 and resource["bytes"] > 0
            if resource["redistribution"] != "allowed":
                assert not resource["auto_download"]
    print("PASS: catalog digests, declarations, code-only packages and licensing metadata")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path)
    validate(parser.parse_args().catalog)
