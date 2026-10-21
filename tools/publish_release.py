"""Publish a tag's audited code package using the workflow-scoped GitHub token."""

import json
import os
import urllib.request
from pathlib import Path

from tools.build_module import build
from tools.prepare_catalog import prepare
from tools.validate_catalog import validate


def request(url, data, content_type="application/json"):
    headers = {
        "Authorization": "Bearer " + os.environ["GITHUB_TOKEN"],
        "Accept": "application/vnd.github+json",
        "Content-Type": content_type,
        "User-Agent": "ObliviousMic-Modules-release",
    }
    with urllib.request.urlopen(
        urllib.request.Request(url, data=data, headers=headers, method="POST"), timeout=60
    ) as response:
        return json.loads(response.read())


def audited_package(tag):
    """Resolve a supported tag before any authenticated publication request."""
    validate()
    name, version = tag.rsplit("-v", 1)
    candidate = json.loads(prepare().read_text("utf-8"))
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
    package = build(name)
    if any(expected[key] != package[key] for key in ("sha256", "bytes", "module_id", "version")):
        raise ValueError("release package differs from the audited catalog candidate")
    return name, version, package


if __name__ == "__main__":
    tag = os.environ["GITHUB_REF_NAME"]
    name, version, package = audited_package(tag)
    release = request(
        "https://api.github.com/repos/Labiey/ObliviousMic-Modules/releases",
        json.dumps(
            {
                "tag_name": tag,
                "name": name + " " + version,
                "body": (
                    "Official code-only module. "
                    "Read the packaged LICENSE, NOTICE, README and repository licensing audit. "
                    "No checkpoint weights or private voice assets are distributed. SHA-256: "
                )
                + package["sha256"],
                "draft": False,
                "prerelease": False,
            }
        ).encode(),
    )
    artifact = Path(package["bundle_path"])
    request(
        release["upload_url"].split("{")[0] + "?name=" + artifact.name,
        artifact.read_bytes(),
        "application/zip",
    )
