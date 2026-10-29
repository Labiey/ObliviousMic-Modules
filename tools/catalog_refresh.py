"""Collect complete community snapshots without executing third-party repository content."""

import argparse
import json
import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from jsonschema import ValidationError

from .catalog_network import ENTRY_NAMES, REPOSITORY, REPOSITORY_ID, CatalogUnavailable, GitHub
from .catalog_snapshot import DATA_FILES, build, downloads, key, validate_snapshot

ROOT = Path(__file__).resolve().parents[1]


def discover(client, kind, start, end, *, depth=0):
    """Partition GitHub's 1,000-result ceiling, with exact-ID deduplication later."""
    if depth > 30:
        raise CatalogUnavailable("github_partition_limit")
    interval = (
        start.isoformat().replace("+00:00", "Z") + ".." + end.isoformat().replace("+00:00", "Z")
    )
    first = client.search(kind, created=interval)
    total = first["total_count"]
    if total > 1000:
        if end - start <= timedelta(seconds=1):
            raise CatalogUnavailable("github_partition_limit")
        middle = start + (end - start) / 2
        return discover(client, kind, start, middle, depth=depth + 1) + discover(
            client, kind, middle, end, depth=depth + 1
        )
    items = list(first["items"])
    for page in range(2, (total + 99) // 100 + 1):
        response = client.search(kind, page=page, created=interval)
        items += response["items"]
    if len({r["id"] for r in items}) < total:
        raise CatalogUnavailable("github_search_incomplete")
    return items


def previous_snapshot(client):
    try:
        branch = client.request(f"/repos/{REPOSITORY}/commits/catalog-data")
    except CatalogUnavailable as exc:
        if exc.status == 404:
            return None
        raise
    repo = {"full_name": REPOSITORY}
    docs = {
        name: client.document(repo, name, branch["sha"], max_size=20 * 1024 * 1024)
        for name in DATA_FILES
    }
    return validate_snapshot(docs)


def collect(client, previous=None, *, now=None, exclusions=None, official=None):
    timestamp = now or datetime.now(UTC)
    stamp = timestamp.isoformat()
    exclusions = exclusions or {"repository_ids": [], "packages": []}
    previous_entries = (
        []
        if previous is None
        else previous["community-modules.json"]["entries"]
        + previous["community-themes.json"]["entries"]
    )
    old_stats = {} if previous is None else previous["stats/downloads.json"]["packages"]
    repositories = {}
    for kind in ENTRY_NAMES:
        for result in discover(client, kind, datetime(2010, 1, 1, tzinfo=UTC), timestamp):
            repositories[result["id"]] = result["full_name"]
    for entry in previous_entries:
        repositories[entry["source"]["repository_id"]] = entry["source"]["repository"]
    official = official or {"modules": []}
    if official.get("modules"):
        repositories[REPOSITORY_ID] = REPOSITORY
    entries, stats, checks = [], {}, {}
    for repository_id, name in sorted(repositories.items()):
        if repository_id in exclusions.get("repository_ids", []):
            checks[str(repository_id)] = "excluded"
            continue
        prior = [e for e in previous_entries if e["source"]["repository_id"] == repository_id]
        try:
            repo = client.repository(name)
            if repo["id"] != repository_id:
                raise ValueError("repository identity changed")
            commit = client.commit(repo)
            kinds = [k for k in ENTRY_NAMES if "obliviousmic-" + k in repo.get("topics", [])]
            paths = [(kind, ENTRY_NAMES[kind]) for kind in kinds]
            if repository_id == REPOSITORY_ID:
                paths += [("module", item["declaration_path"]) for item in official["modules"]]
            fresh = []
            for kind, path in paths:
                try:
                    entry = client.candidate(repo, kind, path=path, commit=commit)
                    if key(entry) not in exclusions.get("packages", []):
                        fresh.append(entry)
                except (ValueError, ValidationError, KeyError):
                    # A malformed candidate is not a runnable package.
                    continue
                except CatalogUnavailable as exc:
                    if exc.status == 404:
                        continue
                    raise
            # Highest version per source/type/ID; duplicate paths cannot inflate downloads.
            from packaging.version import Version

            unique = {}
            for entry in fresh:
                if key(entry) not in unique or Version(entry["version"]) > Version(
                    unique[key(entry)]["version"]
                ):
                    unique[key(entry)] = entry
            fresh = list(unique.values())
            for entry in fresh:
                old = old_stats.get(key(entry), {})
                assets = dict(old.get("observations", [{}])[-1].get("assets", {}))
                selected = entry["declaration"]["distribution"]
                ids = set(assets) | {str(selected["asset_id"])}
                current = {}
                for asset_id in ids:
                    try:
                        asset = client.request(
                            f"/repos/{repo['full_name']}/releases/assets/{int(asset_id)}"
                        )
                        if (
                            asset.get("id") != int(asset_id)
                            or type(asset.get("download_count")) is not int
                            or asset["download_count"] < 0
                        ):
                            raise ValueError("invalid asset download count")
                        current[asset_id] = {
                            "count": asset["download_count"],
                            "created_at": asset.get("created_at"),
                            "digest": asset.get("digest"),
                        }
                    except CatalogUnavailable as exc:
                        if exc.status == 404:
                            continue
                        raise
                stats[key(entry)] = downloads(old, current, stamp)
                entry["checked_at"] = stamp
            entries.extend(fresh)
            checks[str(repository_id)] = "available" if fresh else "no_valid_declaration"
        except CatalogUnavailable as exc:
            if exc.code in {
                "github_rate_limited",
                "github_request_budget",
                "github_search_incomplete",
            }:
                raise  # Never replace a complete snapshot with a partial scan.
            confirmed = exc.status in (404, 410)
            entries.extend(
                {
                    **e,
                    "check_state": "unavailable" if confirmed else "stale",
                    "check_error": exc.code,
                }
                for e in prior
            )
            for e in prior:
                if key(e) in old_stats:
                    stats[key(e)] = old_stats[key(e)]
            checks[str(repository_id)] = exc.code
        except (ValueError, KeyError):
            checks[str(repository_id)] = "identity_invalid"
    snapshot = build(entries, stats, now=stamp, snapshot_id=str(uuid.uuid4()), checks=checks)
    return validate_snapshot(snapshot)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = GitHub(os.environ.get("GITHUB_TOKEN"), retries=3)
    repo = client.repository(REPOSITORY)
    if repo["id"] != REPOSITORY_ID:
        raise ValueError("official repository ID mismatch")
    previous = previous_snapshot(client)
    exclusions = json.loads((ROOT / "catalog/exclusions.json").read_text("utf-8"))
    official = json.loads((ROOT / "catalog/index.json").read_text("utf-8"))
    snapshot = collect(client, previous, exclusions=exclusions, official=official)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, doc in snapshot.items():
        path = args.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
            "utf-8",
        )
    print("PASS: complete community catalog snapshot collected and validated")


if __name__ == "__main__":
    main()
