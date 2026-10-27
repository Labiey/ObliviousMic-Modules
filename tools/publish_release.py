"""Publish a tag's audited code package using the workflow-scoped GitHub token."""

import json
import os
import urllib.error
import urllib.request
from contextlib import nullcontext
from pathlib import Path

from tools.build_module import build
from tools.prepare_catalog import MODULES, ROOT, prepare


def request(url, data=None, content_type="application/json", *, method=None):
    headers = {
        "Authorization": "Bearer " + os.environ["GITHUB_TOKEN"],
        "Accept": "application/vnd.github+json",
        "Content-Type": content_type,
        "User-Agent": "ObliviousMic-Modules-release",
    }
    if isinstance(data, Path):
        headers["Content-Length"] = str(data.stat().st_size)
    with data.open("rb") if isinstance(data, Path) else nullcontext(data) as body:
        with urllib.request.urlopen(
            urllib.request.Request(
                url,
                data=body,
                headers=headers,
                method=method or ("POST" if data is not None else "GET"),
            ),
            timeout=120,
        ) as response:
            return json.loads(response.read())


def audited_package(tag):
    """Resolve a supported tag before any authenticated publication request."""
    name, version = tag.rsplit("-v", 1)
    if (
        name not in MODULES
        or json.loads((ROOT / "modules" / name / "manifest.json").read_text("utf-8"))[
            "module_version"
        ]
        != version
    ):
        raise ValueError("release tag is not in the audited catalog candidate")
    candidate = json.loads(
        prepare(output=ROOT / "dist" / f"{name}.catalog.json", names=[name]).read_text("utf-8")
    )
    expected = next(
        (
            item
            for item in candidate["modules"]
            if item["name"] == name and item["version"] == version
        ),
        None,
    )
    if expected is None:
        raise ValueError("release tag is not in the audited catalog candidate")
    package = build(name, include_models=bool(expected.get("bundled_resources")))
    if any(expected[key] != package[key] for key in ("sha256", "bytes", "module_id", "version")):
        raise ValueError("release package differs from the audited catalog candidate")
    return name, version, package


def publish(tag):
    name, version, package = audited_package(tag)
    endpoint = "https://api.github.com/repos/Labiey/ObliviousMic-Modules/releases"
    try:
        release = request(endpoint + "/tags/" + tag)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        release = request(
            endpoint,
            json.dumps(
                {
                    "tag_name": tag,
                    "target_commitish": os.environ.get("GITHUB_SHA", "main"),
                    "name": name + " " + version,
                    "body": (
                        "Official module package. Code license: " + package["license"] + ". "
                        "Licensed public weights are included where declared. "
                        "Read the packaged LICENSE, NOTICE, README and repository licensing audit. "
                        "Unresolved BSRNN templates and private voice assets are excluded. "
                        "SHA-256: "
                    )
                    + package["sha256"],
                    "draft": True,
                    "prerelease": False,
                }
            ).encode(),
        )
    artifact = Path(package["bundle_path"])
    asset = next((a for a in release.get("assets", []) if a["name"] == artifact.name), None)
    if asset is None:
        asset = request(
            release["upload_url"].split("{")[0] + "?name=" + artifact.name,
            artifact,
            "application/zip",
        )
    if (
        asset.get("size") != package["bytes"]
        or asset.get("digest") != "sha256:" + package["sha256"]
    ):
        raise ValueError("uploaded Release asset size/hash differs; release remains draft")
    if release["draft"]:
        release = request(
            endpoint + "/" + str(release["id"]),
            json.dumps({"draft": False}).encode(),
            method="PATCH",
        )
    return release["html_url"]


if __name__ == "__main__":
    print(publish(os.environ["GITHUB_REF_NAME"]))
