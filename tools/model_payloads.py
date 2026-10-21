"""Fetch pinned public resources and add only audited, redistributable payloads."""

import hashlib
import json
import re
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(data)
    return result.hexdigest()


def acquire(item, cache, *, download=True):
    key, size = item["sha256"], item["bytes"]
    if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
        raise ValueError("invalid resource digest")
    if type(size) is not int or not 0 < size <= 1024 * 1024 * 1024:
        raise ValueError("invalid resource size")
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / key
    if path.exists():
        if path.stat().st_size != size or digest(path) != key:
            raise ValueError("cached model size/hash mismatch")
        return path
    if not download:
        raise FileNotFoundError("verified model cache is missing: " + key)
    url = urlparse(item["download_url"])
    if url.scheme != "https" or url.username or url.password:
        raise ValueError("model download requires HTTPS without credentials")
    temporary = path.with_suffix(".partial")
    try:
        request = urllib.request.Request(
            item["download_url"], headers={"User-Agent": "ObliviousMic-Modules-build/2"}
        )
        total = 0
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary.open("xb") as target,
        ):
            if urlparse(response.geturl()).scheme != "https":
                raise ValueError("model redirect requires HTTPS")
            while data := response.read(1024 * 1024):
                total += len(data)
                if total > size:
                    raise ValueError("model exceeds declared size")
                target.write(data)
        if total != size or digest(temporary) != key:
            raise ValueError("downloaded model size/hash mismatch")
        temporary.replace(path)
        return path
    finally:
        temporary.unlink(missing_ok=True)


def build_release(name, output=None, *, cache=None, download=True):
    from tools.build_module import ROOT, build

    with tempfile.TemporaryDirectory(prefix="obmic-code-") as stage:
        package = build(name, stage)
        with zipfile.ZipFile(package["bundle_path"]) as archive:
            files = {relative: archive.read(relative) for relative in archive.namelist()}
    output = Path(output or ROOT / "dist")
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{name}-{package['version']}.obmic.zip"
    temporary = path.with_suffix(".partial")
    resources = json.loads(files["resources.json"])
    payloads = {}
    for item in resources["assets"]:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", item["id"]):
            raise ValueError("invalid resource role")
        if item["redistribution"] != "allowed":
            if item["auto_download"]:
                raise ValueError("unverified models cannot be downloaded or redistributed")
            continue
        if item["license"] not in {"MIT", "Apache-2.0"} or "license_file" not in item:
            raise ValueError("model redistribution requires an audited license file")
        if not item["license_file"].startswith("licenses/") or item["license_file"] not in files:
            raise ValueError("model license file is missing")
        extension = {"onnx": "onnx", "binary": "bin", "json": "json", "text": "txt"}[
            item.get("format", "onnx")
        ]
        relative = f"models/{item['id']}.{extension}"
        payloads[relative] = acquire(item, cache or ROOT / "dist/model-cache", download=download)
        item["bundled_path"] = relative
    files["resources.json"] = (json.dumps(resources, indent=2, ensure_ascii=False) + "\n").encode()
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for relative in sorted(files.keys() | payloads.keys()):
                info = zipfile.ZipInfo(relative, (2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                if relative in payloads:
                    with payloads[relative].open("rb") as source, archive.open(info, "w") as target:
                        shutil.copyfileobj(source, target, 1024 * 1024)
                else:
                    archive.writestr(info, files[relative])
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    package.update(
        sha256=digest(path),
        bytes=path.stat().st_size,
        bundle_path=str(path),
        bundled_resources=[
            {key: item[key] for key in ("id", "sha256", "bytes", "license", "source_url")}
            for item in resources["assets"]
            if "bundled_path" in item
        ],
    )
    return package
