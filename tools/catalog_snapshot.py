"""Validate and rank complete catalog snapshots; download counts are not installations."""

import hashlib
import json
import re
from datetime import datetime, timedelta

from jsonschema import Draft202012Validator

from .catalog_network import ENTRY_NAMES, SCHEMAS

DATA_FILES = {
    "community-modules.json",
    "community-themes.json",
    "popular-modules.json",
    "popular-themes.json",
    "stats/downloads.json",
    "status.json",
}


def key(entry):
    return f"{entry['source']['repository_id']}/{entry['package_type']}/{entry['package_id']}"


def downloads(previous, current, now):
    """Use observed per-asset increases; first observation only establishes a baseline."""
    timestamp = datetime.fromisoformat(now.replace("Z", "+00:00"))
    history = list(previous.get("observations", []))
    # Keep the last measurement before the seven-day window, plus later ones.
    floor = timestamp - timedelta(days=7)
    before = [h for h in history if datetime.fromisoformat(h["at"].replace("Z", "+00:00")) <= floor]
    inside = [h for h in history if datetime.fromisoformat(h["at"].replace("Z", "+00:00")) > floor]
    history = (before[-1:] + inside)[-32:]
    history.append({"at": now, "assets": current})
    delta, resets = 0, []
    for left, right in zip(history, history[1:], strict=False):
        if datetime.fromisoformat(left["at"].replace("Z", "+00:00")) < floor:
            continue
        for asset_id, asset in right["assets"].items():
            prior = left["assets"].get(asset_id)
            if prior is None:
                continue
            if (
                asset["count"] < prior["count"]
                or asset.get("created_at") != prior.get("created_at")
                or asset.get("digest") != prior.get("digest")
            ):
                resets.append({"asset_id": asset_id, "at": right["at"]})
            else:
                delta += asset["count"] - prior["count"]
    oldest = datetime.fromisoformat(history[0]["at"].replace("Z", "+00:00"))
    return {
        "observations": history,
        "observed_download_delta": delta,
        "coverage_days": min(7, max(0, (timestamp - oldest).total_seconds() / 86400)),
        "counter_resets": resets,
        "unit": "release_asset_downloads",
    }


def build(entries, stats, *, now, snapshot_id, checks=None):
    result = {}
    for kind, plural in [("module", "modules"), ("theme", "themes")]:
        values = [e for e in entries if e["package_type"] == kind]
        result[f"community-{plural}.json"] = {
            "schema_version": 1,
            "snapshot_id": snapshot_id,
            "generated_at": now,
            "entries": values,
        }
        popular = []
        for entry in values:
            if entry.get("check_state") not in {"available", "stale"}:
                continue
            score = stats.get(key(entry), {})
            popular.append(
                {
                    **entry,
                    "popularity": {
                        k: score.get(k, 0) for k in ["observed_download_delta", "coverage_days"]
                    },
                }
            )
        popular.sort(key=lambda e: (-e["popularity"]["observed_download_delta"], key(e)))
        result[f"popular-{plural}.json"] = {
            "schema_version": 1,
            "snapshot_id": snapshot_id,
            "generated_at": now,
            "entries": popular,
        }
    result["stats/downloads.json"] = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "generated_at": now,
        "packages": stats,
    }
    encoded = {
        name: (
            json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        for name, doc in result.items()
    }
    result["status.json"] = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "generated_at": now,
        "complete": True,
        "checks": checks or {},
        "files": {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            for name, data in encoded.items()
        },
    }
    return result


def validate_snapshot(documents):
    if set(documents) != DATA_FILES:
        raise ValueError("catalog snapshot has unexpected or missing files")
    status = documents["status.json"]
    if status.get("complete") is not True or status.get("schema_version") != 1:
        raise ValueError("catalog snapshot is incomplete")
    datetime.fromisoformat(status["generated_at"].replace("Z", "+00:00"))
    if not isinstance(status.get("snapshot_id"), str) or len(status["snapshot_id"]) > 128:
        raise ValueError("invalid snapshot ID")
    for name in DATA_FILES - {"status.json"}:
        doc = documents[name]
        if (
            doc.get("snapshot_id") != status["snapshot_id"]
            or doc.get("schema_version") != 1
            or doc.get("generated_at") != status["generated_at"]
        ):
            raise ValueError("mixed catalog snapshot revisions")
        data = (
            json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        if len(data) > 20 * 1024 * 1024 or status["files"].get(name) != {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }:
            raise ValueError("catalog snapshot digest mismatch")
        if name.startswith(("popular-", "community-")):
            entries = doc.get("entries")
            if not isinstance(entries, list) or len(entries) > 10000:
                raise ValueError("catalog snapshot entry limit")
            seen = set()
            for entry in entries:
                if entry.get("package_type") not in ENTRY_NAMES or key(entry) in seen:
                    raise ValueError("invalid or duplicate catalog entry")
                if (
                    type(entry["source"].get("repository_id")) is not int
                    or entry["source"]["repository_id"] < 1
                ):
                    raise ValueError("invalid catalog source")
                if (
                    entry["declaration"]["distribution"]["repository_id"]
                    != entry["source"]["repository_id"]
                ):
                    raise ValueError("catalog source mismatch")
                kind = entry["package_type"]
                declaration = entry["declaration"]
                schema = json.loads((SCHEMAS / f"obliviousmic-{kind}-v1.json").read_text("utf-8"))
                Draft202012Validator(schema).validate(declaration)
                source = entry["source"]
                distribution = declaration["distribution"]
                if (
                    declaration["package_type"] != kind
                    or declaration["module_id" if kind == "module" else "id"] != entry["package_id"]
                    or declaration["module_version" if kind == "module" else "version"]
                    != entry["version"]
                    or not re.fullmatch(r"[0-9a-f]{40}", source.get("commit", ""))
                    or source["repository"].split("/")[0] != source["owner"]
                    or declaration["repository"] != "https://github.com/" + source["repository"]
                    or distribution["release_id"] != source["release_id"]
                    or distribution["asset_id"] != source["asset_id"]
                ):
                    raise ValueError("catalog declaration identity mismatch")
                seen.add(key(entry))
    return documents
