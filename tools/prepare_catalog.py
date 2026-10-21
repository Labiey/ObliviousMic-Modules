"""Build a reviewable candidate without changing the published catalog."""

import argparse
import json
from pathlib import Path

from tools.build_module import ROOT, build

NEW_MODULES = {
    "asr": (
        "Local speech recognition with timestamped Chinese, English and Japanese text",
        "MIT Whisper conversion; pinned resources downloaded separately",
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
}


def prepare(output=None):
    output = Path(output or ROOT / "dist" / "catalog.next.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((ROOT / "catalog/index.json").read_text("utf-8"))
    existing = {item["name"] for item in catalog["modules"]}
    for name, (description, weight_status) in NEW_MODULES.items():
        if name in existing:
            continue
        package = build(name, output.parent)
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
    args = parser.parse_args()
    print(prepare(args.output))
