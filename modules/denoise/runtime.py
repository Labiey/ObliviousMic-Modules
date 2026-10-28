# SPDX-License-Identifier: Apache-2.0
"""Module API v1 adapter for local DPDFNet speech denoising."""

import json
from fractions import Fraction
from pathlib import Path

from obliviousmic.sdk import FlushResult, Latency, PortBundle

from .engine import LATENCY_FRAMES, RATE, Session, Stream

ROOT = Path(__file__).parent


class Module:
    MANIFEST = json.loads((ROOT / "manifest.json").read_text("utf-8"))
    SCHEMA = json.loads((ROOT / "settings.schema.json").read_text("utf-8"))

    def __init__(self, factory=None):
        self.factory = factory
        self.lease = self.processor = None

    def describe(self):
        return self.MANIFEST

    def prepare(self, context, config):
        self.dispose()
        self.context, self.config = context, dict(config)
        mode = config["mode"]
        if self.factory:
            session = self.factory(mode, config["threads"])
        else:
            path = Path(context.capabilities["asset.read"](config[mode + "_model_asset"]))
            self.lease = context.capabilities["resource.acquire"](
                path,
                {"engine": "denoise-dpdfnet-1", "device": "CPU", "threads": config["threads"]},
                lambda: Session(path, mode, config["threads"]),
            )
            session = self.lease.value
        try:
            self.processor = Stream(session)
            self.reset("prepare")
        except Exception:
            self.dispose()
            raise

    def start(self, stream):
        self.stream = stream
        self.reset("start")

    def reset(self, reason):
        if self.processor:
            self.processor.reset()
        self.first = self.expected_end = None
        self.emitted = self.sequence = 0

    def _output(self, samples):
        if not len(samples):
            return ()
        packet = self.first.with_samples(
            samples[:, None],
            media_start=self.first.media_start + Fraction(self.emitted, RATE),
            sequence=self.sequence,
            flags=frozenset(),
        )
        self.emitted += len(samples)
        self.sequence += 1
        return (packet,)

    def process(self, bundle):
        result = []
        for packet in bundle.ports.get("audio", ()):
            if packet.format.sample_rate != RATE or packet.format.channels != 1:
                raise ValueError("DPDFNet requires negotiated 48000 Hz mono PCM")
            if self.first is not None and (
                packet.stream_id != self.first.stream_id
                or packet.epoch_id != self.first.epoch_id
                or packet.media_start != self.expected_end
                or "discontinuity" in packet.flags
            ):
                # Never join a previous stream's recurrent/PCM state to a new epoch.
                self.reset("discontinuity")
            if self.first is None:
                self.first = packet
            self.expected_end = packet.media_end
            result.extend(
                self._output(self.processor.process(packet.samples[:, 0], self.config["mix"]))
            )
        return PortBundle({"audio": tuple(result)})

    def flush(self):
        if self.first is None:
            return FlushResult()
        return FlushResult(
            PortBundle({"audio": self._output(self.processor.flush(self.config["mix"]))})
        )

    def report_latency(self):
        return Latency(Fraction(LATENCY_FRAMES, RATE))

    def report_monitor(self):
        return {
            "enhanced": self.config["mode"] == "enhanced",
            "threads": self.config["threads"],
            "mix": self.config["mix"],
        }

    def update_parameters(self, change):
        if set(change) - {"mix"}:
            raise ValueError("mode and thread changes require reprepare")
        if "mix" in change:
            value = change["mix"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value <= 1
            ):
                raise ValueError("mix must be between 0 and 1")
            self.config["mix"] = float(value)

    def stop(self):
        self.reset("stop")

    def dispose(self):
        self.processor = None
        self.first = self.expected_end = None
        if self.lease:
            self.lease.release()
            self.lease = None
