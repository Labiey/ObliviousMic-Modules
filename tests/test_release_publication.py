import hashlib
import urllib.error
from pathlib import Path

import pytest

from tools import publish_release


@pytest.mark.parametrize("existing,corrupt", [(False, False), (True, False), (False, True)])
def test_publish_checks_uploaded_digest_before_publication_and_never_replaces_assets(
    tmp_path, monkeypatch, existing, corrupt
):
    path = tmp_path / "fixture.obmic.zip"
    path.write_bytes(b"fixture")
    package = {
        "bundle_path": str(path),
        "bytes": 7,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    asset = {"name": path.name, "size": 7, "digest": "sha256:" + package["sha256"]}
    release = {
        "id": 12,
        "draft": not existing,
        "assets": [asset] if existing else [],
        "upload_url": "https://uploads.github.com/release{?name}",
        "html_url": "https://github.com/example/release",
    }
    monkeypatch.setattr(publish_release, "audited_package", lambda _: ("fixture", "1.0", package))
    calls = []

    def request(url, data=None, *_args, **kwargs):
        calls.append((url, data, kwargs))
        if "/tags/" in url:
            if existing:
                return release
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)
        if url.startswith("https://uploads."):
            assert isinstance(data, Path)  # Large uploads stream, never read into one byte string.
            return {**asset, "digest": "sha256:wrong"} if corrupt else asset
        return release

    monkeypatch.setattr(publish_release, "request", request)
    if corrupt:
        with pytest.raises(ValueError, match="size/hash"):
            publish_release.publish("fixture-v1.0")
        assert not any(c[2].get("method") == "PATCH" for c in calls)
    else:
        assert publish_release.publish("fixture-v1.0") == release["html_url"]
        assert len(calls) == (1 if existing else 4)
