"""Validate published declarations and deterministic package digests without executing code."""

import argparse
import ast
import json
import tempfile
from pathlib import Path

from tools.build_module import build
from tools.catalog_network import REPOSITORY_ID, ENTRY_NAMES
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]


def validate(path=None, *, metadata_only=False):
    catalog = json.loads(Path(path or ROOT / "catalog/index.json").read_text("utf-8"))
    assert (
        catalog["catalog_version"] == 1 and catalog["repository"] == "Labiey/ObliviousMic-Modules"
    )
    ids = [item["module_id"] for item in catalog["modules"]]
    assert len(ids) == len(set(ids)), "duplicate catalog module"
    for item in catalog["modules"]:
        name = item["name"]
        folder = ROOT / "modules" / name
        if item.get("declaration", {}).get("distribution"):
            declared = item["declaration"]
            schema = json.loads((ROOT / "schemas/obliviousmic-module-v1.json").read_text("utf-8"))
            Draft202012Validator(schema).validate(declared)
            internal = json.loads((folder / ENTRY_NAMES["module"]).read_text("utf-8"))
            assert {k: v for k, v in declared.items() if k != "distribution"} == internal
            distribution = declared["distribution"]
            assert catalog["repository_id"] == distribution["repository_id"] == REPOSITORY_ID
            assert (distribution["bytes"], distribution["sha256"], distribution["url"]) == (item["bytes"], item["sha256"], item["download_url"])
            declaration_path = ROOT / item["declaration_path"]
            if Path(path or ROOT / "catalog/index.json").resolve() == (ROOT / "catalog/index.json").resolve():
                assert json.loads(declaration_path.read_text("utf-8")) == declared
        assert name in {
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
        assert len(item["sha256"]) == 64 and 0 < item["bytes"] <= 1024 * 1024 * 1024
        if not metadata_only:
            with tempfile.TemporaryDirectory(prefix="obmic-catalog-") as stage:
                package = build(name, stage, include_models=bool(item.get("bundled_resources")))
            assert item["sha256"] == package["sha256"] and item["bytes"] == package["bytes"], (
                f"{name}: catalog {item['sha256']} / {item['bytes']} bytes; "
                f"built {package['sha256']} / {package['bytes']} bytes"
            )
            assert (
                item["module_id"] == package["module_id"] and item["version"] == package["version"]
            )
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
    print("PASS: catalog declarations, package digests and licensing metadata")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    validate(args.catalog, metadata_only=args.metadata_only)
