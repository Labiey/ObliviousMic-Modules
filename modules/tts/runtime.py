# SPDX-License-Identifier: Apache-2.0
"""Private reference binding and bounded network-worker ownership."""

import json
import sys
from pathlib import Path

from obliviousmic.adapters.process import ProcessModule
from obliviousmic.sdk import PrepareContext

from . import local_service
from .providers import TTSError, validate_config

ROOT = Path(__file__).parent
MODULE_ID = "org.obliviousmic.tts"


def initialize_reference(ctx, args):
    import soundfile as sf

    samples = ctx.capabilities["audio.read"](args["file_id"])
    if not 3 * 16000 <= len(samples) <= 10 * 16000:
        raise ValueError("tts_reference_audio_requires_3_to_10_seconds")
    path = ctx.workdir / "reference.wav"
    sf.write(path, samples, 16000, subtype="PCM_16")
    artifact = ctx.job.commit(
        lambda: ctx.capabilities["profile.write"](
            args["profile_id"],
            MODULE_ID,
            "reference_audio",
            path,
            {
                "prompt_text": args.get("prompt_text", ""),
                "prompt_language": args.get("prompt_language", "zh"),
                "sample_rate": 16000,
                "channels": 1,
            },
            "user-selected-reference",
            "pcm16k-reference-v1",
        )
    )
    return {
        "profile_id": args["profile_id"],
        "artifact_kind": "reference_audio",
        "sha256": artifact["sha256"],
    }


def worker_command():
    import obliviousmic

    paths = list(sys.path) + [str(Path(obliviousmic.__file__).resolve().parent.parent)]
    code = (
        "import sys;sys.path[:0]="
        + repr(paths)
        + ";from obliviousmic.adapters.worker import main;main()"
    )
    return [
        getattr(sys, "_base_executable", sys.executable),
        "-B",
        "-c",
        code,
        "--root",
        str(ROOT),
        "--entrypoint",
        "network:Module",
    ]


class Module:
    MANIFEST = json.loads((ROOT / "manifest.json").read_text("utf-8"))
    SCHEMA = json.loads((ROOT / "settings.schema.json").read_text("utf-8"))

    def __init__(self):
        self.worker = self.service = None

    def describe(self):
        return self.MANIFEST

    def prepare(self, context, config):
        self.dispose()
        config = dict(config)
        validate_config(config)
        try:
            if config["provider"] == "gpt_sovits":
                if not config["reference_path"]:
                    if not config["profile_id"]:
                        raise TTSError("tts_reference_audio_required")
                    read = context.capabilities["profile.read"]
                    try:
                        config["reference_path"] = str(
                            read(config["profile_id"], "reference_audio", MODULE_ID)
                        )
                        info = read(
                            config["profile_id"], "reference_audio", MODULE_ID, metadata=True
                        )["compatibility"]
                    except (KeyError, ValueError, FileNotFoundError):
                        raise TTSError("tts_reference_audio_required") from None
                    if not config["reference_prompt"]:
                        config["reference_prompt"] = info.get("prompt_text", "")
                    if config["reference_language"] == "auto":
                        config["reference_language"] = info.get("prompt_language", "zh")
                if not Path(config["reference_path"]).is_file():
                    raise TTSError("tts_reference_audio_missing")
                self.service = local_service.acquire(
                    config, context.capabilities["service.acquire"]
                )
            self.worker = ProcessModule(
                worker_command(),
                timeout=93
                if config["provider"] == "gpt_sovits"
                else config["request_timeout_seconds"] + 3,
            )
            isolated = PrepareContext(
                context.node_instance_id, context.inputs, context.outputs, context.mode
            )
            # Credentials and private paths travel only over bounded IPC, never command arguments.
            self.worker.prepare(isolated, config)
            self.worker.timeout = config["request_timeout_seconds"] + 3
        except BaseException:
            self.dispose()
            raise

    def start(self, stream):
        self.worker.start(stream)

    def process(self, bundle):
        return self.worker.process(bundle)

    def update_parameters(self, change):
        raise ValueError("TTS provider changes require reprepare")

    def report_latency(self):
        return self.worker.report_latency()

    def flush(self):
        return self.worker.flush()

    def reset(self, reason):
        self.worker.reset(reason)

    def stop(self):
        self.worker.stop()

    def dispose(self):
        if self.worker:
            self.worker.dispose()
            self.worker = None
        if self.service:
            self.service.release()
            self.service = None
