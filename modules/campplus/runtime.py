# SPDX-License-Identifier: Apache-2.0
# Split from ObliviousMic's adapted frozen OwnerMic leaves; see NOTICE.
import json
from fractions import Fraction
from pathlib import Path
from time import monotonic

import numpy as np
from obliviousmic.sdk import FlushResult, Latency, MetadataPacket, PortBundle
from obliviousmic.sdk.declaration import audio, data_port, manifest, parameter, settings

from .actions import initialize_action as initialize_action
from .actions import validate_action as validate_action
from .lib.base import COMMON, ModelModule, model_path, session


class Module(ModelModule):
    MANIFEST = manifest(
        "org.obliviousmic.campplus",
        [audio(rates=[16000], channels=[1])],
        [data_port("decision")],
        preferred_block_frames=160,
        minimum_block_frames=160,
        exact_block_frames=160,
        allow_partial_eos=True,
        async_process=True,
        max_tail_frames=0,
    )
    MANIFEST["permissions"] = ["asset.read", "profile.read", "resource.acquire"]
    MANIFEST["runtime"]["entrypoint"] = "runtime:Module"
    MANIFEST["module_version"] = "1.1.0"
    MANIFEST["dependencies"] = ["onnx>=1.16"]
    SCHEMA = settings(
        {
            **COMMON,
            "embedding_path": parameter("string", "", private=True),
            "window_ms": parameter("integer", 600, minimum=200, maximum=3000, multipleOf=10),
            "hop_ms": parameter("integer", 100, minimum=10, maximum=1000, multipleOf=10),
            "veto_window_ms": parameter("integer", 600, minimum=200, maximum=3000, multipleOf=10),
            "silence_floor_dbfs": parameter("number", -55, minimum=-100, maximum=0),
        }
    )

    def prepare(self, context, config):
        from .lib.identity_stream import SlidingIdentityProvider, WindowEmbeddingExtractor

        self.context, self.config = context, dict(config)
        if self.factory:
            self.processor = self.factory(context, config)
        else:
            path = model_path(context, config)
            raw, self.lease = session(context, config, path)
            extractor = WindowEmbeddingExtractor(
                path,
                engine=config["engine"],
                device=config["device"],
                num_threads=config["threads"],
                probe_window_samples=config["window_ms"] * 16,
                session_factory=lambda *_: raw,
            )
            extractor.initialize()

            class Checked:
                def embed(self, audio):
                    value = extractor.embed(audio)
                    if value is None:
                        raise ValueError("identity inference produced no valid embedding")
                    return value

            if config["profile_id"]:
                embedding = context.capabilities["profile.read"](config["profile_id"], "embedding")
            else:
                embedding = np.load(config["embedding_path"], allow_pickle=False)
            self.processor = SlidingIdentityProvider(
                embedding,
                Checked(),
                window_ms=config["window_ms"],
                hop_ms=config["hop_ms"],
                veto_window_ms=config["veto_window_ms"],
                silence_floor_dbfs=config["silence_floor_dbfs"],
            )
        self.processor.initialize()
        self.reset("prepare")

    def reset(self, reason):
        if getattr(self, "processor", None):
            self.processor.reset()
        self.frames_seen = 0
        self.origin = None

    def process(self, bundle):
        outputs = []
        for p in bundle.ports.get("audio", ()):
            if self.origin is None:
                self.origin = p.media_start
            frame = (
                np.pad(p.samples[:, 0], (0, 160 - p.frames)) if p.frames < 160 else p.samples[:, 0]
            )
            observed = self.processor.observe(frame, sample_rate=16000)
            self.frames_seen += 160
            window = self.config["window_ms"] * 16
            hop = self.config["hop_ms"] * 16
            window_end = (
                ((self.frames_seen - window) // hop) * hop + window
                if self.frames_seen >= window
                else self.frames_seen
            )
            start = self.origin + Fraction(max(0, window_end - window), 16000)
            end = min(p.media_end, self.origin + Fraction(window_end, 16000))

            outputs.append(
                MetadataPacket(
                    start,
                    end,
                    monotonic(),
                    1,
                    self.context.node_instance_id,
                    {
                        "score": observed.score,
                        "veto": observed.veto,
                        "mixture_energy_db": observed.mixture_energy_db,
                        "effective_start": [p.media_start.numerator, p.media_start.denominator],
                        "effective_end": [p.media_end.numerator, p.media_end.denominator],
                    },
                    p.stream_id,
                    p.epoch_id,
                    private=True,
                )
            )
        return PortBundle({"decision": tuple(outputs)})

    def flush(self):
        return FlushResult()

    def report_latency(self):
        return Latency(Fraction(self.config["window_ms"] + self.config["hop_ms"], 1000))


# Package declarations are the source of truth for both bundled and source use.
Module.MANIFEST = json.loads(Path(__file__).with_name("manifest.json").read_text("utf-8"))
Module.SCHEMA = json.loads(Path(__file__).with_name("settings.schema.json").read_text("utf-8"))
