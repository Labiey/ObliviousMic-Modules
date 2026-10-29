"""Publish only validated generated data to catalog-data, never source or identity catalogs."""

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from .catalog_network import REPOSITORY, REPOSITORY_ID, CatalogUnavailable, GitHub
from .catalog_snapshot import DATA_FILES, validate_snapshot


def load(folder):
    root = Path(folder).resolve()
    names = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if names != DATA_FILES:
        raise ValueError("artifact contains unexpected or missing files")
    documents = {}
    for name in DATA_FILES:
        path = root / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root)
            or path.stat().st_size > 20 * 1024 * 1024
        ):
            raise ValueError("invalid snapshot artifact path/size")
        documents[name] = json.loads(path.read_text("utf-8"))
    return validate_snapshot(documents)


class Writer(GitHub):
    def write(self, path, payload, *, method="POST"):
        if not self.token or not path.startswith(f"/repos/{REPOSITORY}/git/"):
            raise ValueError("publication is limited to the official catalog Git API")
        request = urllib.request.Request(
            "https://api.github.com" + path,
            data=json.dumps(payload).encode(),
            method=method,
            headers={
                "Authorization": "Bearer " + self.token,
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "User-Agent": "ObliviousMic-Catalog/1",
            },
        )
        try:
            with self.opener(request, timeout=60) as response:
                if response.geturl().split("/", 3)[2] != "api.github.com":
                    raise ValueError("publication redirect is forbidden")
                return json.loads(response.read(2 * 1024 * 1024))
        except urllib.error.HTTPError:
            raise CatalogUnavailable("catalog_publication_failed") from None


def publish(client, documents):
    validate_snapshot(documents)
    repo = client.repository(REPOSITORY)
    if repo["id"] != REPOSITORY_ID:
        raise ValueError("official repository ID mismatch")
    base = f"/repos/{REPOSITORY}/git/"
    parent = None
    try:
        parent = client.request(base + "ref/heads/catalog-data")["object"]["sha"]
        old = client.request(base + "trees/" + parent, {"recursive": "1"})
        if old.get("truncated") or any(
            e["path"] not in DATA_FILES for e in old["tree"] if e["type"] == "blob"
        ):
            raise ValueError("catalog-data contains files outside generated-data scope")
    except CatalogUnavailable as exc:
        if exc.status != 404:
            raise
    tree = [
        {
            "path": name,
            "mode": "100644",
            "type": "blob",
            "content": json.dumps(
                documents[name], ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            + "\n",
        }
        for name in sorted(DATA_FILES)
    ]
    created = client.write(base + "trees", {"tree": tree})
    commit = client.write(
        base + "commits",
        {
            "message": "Refresh community catalog " + documents["status.json"]["generated_at"],
            "tree": created["sha"],
            "parents": [parent] if parent else [],
        },
    )
    if parent:
        client.write(
            base + "refs/heads/catalog-data", {"sha": commit["sha"], "force": False}, method="PATCH"
        )
    else:
        client.write(base + "refs", {"ref": "refs/heads/catalog-data", "sha": commit["sha"]})
    return commit["sha"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    result = publish(Writer(os.environ.get("GITHUB_TOKEN")), load(args.input))
    print("Published complete catalog snapshot commit " + result)
