"""Build a reviewable candidate without changing the published catalog."""

import argparse
import json
from pathlib import Path

from tools.build_module import ROOT, build

MODULES = {
    "campplus": (
        "Speaker identity with one-time registration",
        "Includes the Apache-2.0 public CAM++ model; private enrollment stays local",
    ),
    "bsrnn": (
        "Target-speaker extraction; authorized local template required",
        "Includes CAM++ speaker weights; separator template rights unresolved, local import only",
    ),
    "asr": (
        "Local speech recognition with timestamped Chinese, English and Japanese text",
        "Includes pinned MIT Whisper Small weights, tokenizer and configuration",
    ),
    "tts": (
        "Text to speech with Qwen, Azure or local GPT-SoVITS",
        "Original adapter only; provider and selected local model terms remain separate",
    ),
    "subtitles": (
        "Private timestamped text in a Windows floating subtitle window",
        "No model resources; Python/Tk runtime required",
    ),
    "broadcast": (
        "Buffered NVIDIA Broadcast enhancement of the graph's upstream audio",
        "No NVIDIA binaries or models; compatible installed Broadcast app required",
    ),
    "translate": (
        "Text translation with LunaTranslator web/API engines and OpenAI-compatible services",
        "GPL-3.0 module with pinned LunaTranslator sources and notices; no model weights",
    ),
    "denoise": (
        "Local DPDFNet speech denoising with Light and Enhanced 48 kHz modes",
        "Includes both pinned Apache-2.0 DPDFNet ONNX models; no voice registration required",
    ),
}


def prepare(output=None, *, include_models=True, cache=None, download=True, names=None):
    output = Path(output or ROOT / "dist" / "catalog.next.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    catalog = {"catalog_version": 1, "repository": "Labiey/ObliviousMic-Modules", "modules": []}
    for name, (description, weight_status) in MODULES.items():
        if names is not None and name not in names:
            continue
        package = build(
            name, output.parent, include_models=include_models, cache=cache, download=download
        )
        package.pop("bundle_path")
        repository = "https://github.com/" + catalog["repository"]
        package.update(
            minimum_framework_version="0.3.0rc1",
            description=description,
            download_url=(
                f"{repository}/releases/download/{name}-v{package['version']}/"
                f"{name}-{package['version']}.obmic.zip"
            ),
            license_url=f"{repository}/blob/main/modules/{name}/LICENSE",
            weight_status=weight_status,
        )
        catalog["modules"].append(package)
    output.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--code-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    print(
        prepare(
            args.output,
            include_models=not args.code_only,
            cache=args.cache,
            download=not args.offline,
        )
    )
