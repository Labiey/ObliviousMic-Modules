"""Build deterministic code packages or licensed, model-inclusive release packages."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(name, output=None, *, include_models=False, cache=None, download=True):
    if include_models:
        from tools.model_payloads import build_release

        return build_release(name, output, cache=cache, download=download)
    folder = ROOT / "modules" / name
    manifest = json.loads((folder / "manifest.json").read_text("utf-8"))
    output = Path(output or ROOT / "dist")
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{name}-{manifest['module_version']}.obmic.zip"
    files = sorted(
        (
            p
            for p in folder.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and "tests" not in p.parts
        ),
        key=lambda p: p.relative_to(folder).as_posix(),
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for file in files:
            if file.suffix not in {".py", ".json", ".md", ".txt"} and file.name not in {
                "LICENSE",
                "NOTICE",
            }:
                raise ValueError("code-only module contains prohibited file")
            info = zipfile.ZipInfo(file.relative_to(folder).as_posix(), (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            # Git checkouts may use CRLF on Windows. Publish identical UTF-8/LF
            # packages on every platform, independent of local checkout settings.
            content = file.read_text("utf-8-sig").replace("\r\n", "\n")
            archive.writestr(info, content.encode("utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "module_id": manifest["module_id"],
        "name": name,
        "version": manifest["module_version"],
        "api": manifest["api"],
        "license": "GPL-3.0-only" if name == "translate" else "Apache-2.0",
        "bytes": path.stat().st_size,
        "sha256": digest,
        "source_path": "modules/" + name,
        "bundle_path": str(path),
        "bundled_resources": [],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name",
        choices=[
            "campplus",
            "bsrnn",
            "asr",
            "asr_api",
            "tts",
            "subtitles",
            "broadcast",
            "translate",
            "denoise",
        ],
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--code-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.name,
                args.output,
                include_models=not args.code_only,
                cache=args.cache,
                download=not args.offline,
            ),
            indent=2,
        )
    )
