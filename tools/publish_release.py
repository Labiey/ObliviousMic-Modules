"""Publish a tag's audited code package using the workflow-scoped GitHub token."""

import json
import os
import urllib.request
from pathlib import Path

from tools.build_module import build
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


if __name__ == "__main__":
    validate()
    tag = os.environ["GITHUB_REF_NAME"]
    name, version = tag.rsplit("-v", 1)
    package = build(name)
    if version != package["version"]:
        raise ValueError("release tag version differs from manifest")
    release = request(
        "https://api.github.com/repos/Labiey/ObliviousMic-Modules/releases",
        json.dumps(
            {
                "tag_name": tag,
                "name": name + " " + version,
                "body": (
                    "Official code-only module. "
                    "Read the packaged LICENSE, NOTICE and licensing.md. "
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
