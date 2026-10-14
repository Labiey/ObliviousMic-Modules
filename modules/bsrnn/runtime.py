# SPDX-License-Identifier: Apache-2.0
# Split from ObliviousMic's adapted frozen OwnerMic leaves; see NOTICE.
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from obliviousmic.sdk import FlushResult, Latency, PortBundle
from obliviousmic.sdk.declaration import audio, manifest, parameter, settings

from .actions import initialize_action as initialize_action
from .actions import validate_action as validate_action
from .lib.base import COMMON, ModelModule, model_path, session


class Module(ModelModule):
    MANIFEST = manifest(
        "org.obliviousmic.bsrnn",
        [audio("mixture", rates=[16000], channels=[1])],
        [audio("separated", rates=[16000], channels=[1])],
        preferred_block_frames=640,
        max_tail_frames=4096,
    )
    MANIFEST["permissions"] = ["asset.read", "profile.read", "resource.acquire"]
    MANIFEST["runtime"]["entrypoint"] = "runtime:Module"
    MANIFEST["module_version"] = "1.1.0"
    MANIFEST["dependencies"] = ["onnx>=1.16"]
    SCHEMA = settings(
        {
            **COMMON,
            "win": parameter("integer", 512, minimum=2, maximum=2048, multipleOf=2),
            "hop": parameter("integer", 128, minimum=1, maximum=512),
        }
    )

    def prepare(self, context, config):
        from .lib.frame_glue import NumpyFrameStream
        from .lib.onnx_frame_backend import OnnxFrameBackend

        self.context, self.config = context, dict(config)
        if self.factory:
            backend = self.factory(context, config)
        else:
            path = model_path(context, config, private=True)
            raw, self.lease = session(context, config, path)
            backend = OnnxFrameBackend(session=raw, win=config["win"], hop=config["hop"])
        self.processor = NumpyFrameStream(backend=backend, win=config["win"], hop=config["hop"])
        self.reset("prepare")

    def reset(self, reason):
        if getattr(self, "processor", None):
            self.processor.reset()
        self.first = None
        self.emitted = 0

    def _output(self, samples):
        if not len(samples):
            return PortBundle()
        packet = self.first.with_samples(
            samples[:, None],
            media_start=self.first.media_start + Fraction(self.emitted, 16000),
            sequence=self.emitted,
        )
        self.emitted += len(samples)
        return PortBundle.audio(packet, "separated")

    def process(self, bundle):
        outputs = []
        for p in bundle.ports.get("mixture", ()):
            if self.first is None:
                self.first = p
            outputs.extend(
                self._output(self.processor.process(p.samples[:, 0])).ports.get("separated", ())
            )
        return PortBundle({"separated": tuple(outputs)})

    def flush(self):
        if self.first is None:
            return FlushResult()
        complete = self.processor.flush()
        # The frozen stream emits a hop-aligned prefix. API v1 also drains the already
        # reconstructed, bounded OLA residual through the exact input end, without
        # padding the model input or modifying any prefix sample/DSP implementation.
        residual = self.processor._emit(
            self.processor.geometry.half + self.processor.processed_samples
        )
        return FlushResult(self._output(np.concatenate((complete, residual))))

    def report_latency(self):
        return Latency(Fraction(self.processor.latency_samples, 16000))


# Package declarations are the source of truth for both bundled and source use.
Module.MANIFEST = json.loads(Path(__file__).with_name("manifest.json").read_text("utf-8"))
Module.SCHEMA = json.loads(Path(__file__).with_name("settings.schema.json").read_text("utf-8"))

