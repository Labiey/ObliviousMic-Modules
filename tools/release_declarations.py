"""Bind audited package candidates to verified, manually published GitHub assets."""

import copy
import json
import re
from pathlib import Path

from jsonschema import Draft202012Validator

from tools.catalog_network import REPOSITORY, REPOSITORY_ID


def bind(candidate, releases, source_commit):
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("invalid source commit")
    result = copy.deepcopy(candidate)
    if result.get("repository_id") != REPOSITORY_ID or result["repository"] != REPOSITORY:
        raise ValueError("official repository identity mismatch")
    schema = json.loads((Path(__file__).parents[1] / "schemas/obliviousmic-module-v1.json").read_text("utf-8"))
    for item in result["modules"]:
        tag = f"{item['name']}-v{item['version']}"
        release = releases[tag]
        if release.get("draft") or release["tag_name"] != tag or type(release["id"]) is not int:
            raise ValueError("release is not a published matching tag")
        assets = [a for a in release["assets"] if a["browser_download_url"] == item["download_url"]]
        if len(assets) != 1:
            raise ValueError("release must contain the exact installer asset")
        asset = assets[0]
        if asset["size"] != item["bytes"] or asset.get("digest") != "sha256:" + item["sha256"]:
            raise ValueError("published installer size/digest mismatch")
        document = item["declaration"]
        if "distribution" in document:
            raise ValueError("candidate must contain the internal declaration")
        document["distribution"] = {
            "repository_id": REPOSITORY_ID,
            "release_id": release["id"],
            "asset_id": asset["id"],
            "release": tag,
            "url": item["download_url"],
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        }
        Draft202012Validator(schema).validate(document)
        item["source_commit"] = source_commit
    return result


def write(catalog, root):
    """An explicit manual promotion; generated catalog-data jobs never call this."""
    root = Path(root)
    for item in catalog["modules"]:
        path = root / item["declaration_path"]
        expected = root / "catalog/declarations" / item["name"] / "obliviousmic.module.json"
        if path.resolve() != expected.resolve():
            raise ValueError("unexpected official declaration path")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(item["declaration"], indent=2, ensure_ascii=False) + "\n", "utf-8")
    (root / "catalog/index.json").write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", "utf-8")
