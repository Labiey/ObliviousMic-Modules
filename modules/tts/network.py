# SPDX-License-Identifier: Apache-2.0
"""Isolated synthesis worker; its clock counts generated audio, not source duration."""

import json
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path

from obliviousmic.sdk import AudioPacket, FlushResult, Latency, PortBundle, TextSegment

try:
    from .providers import Client
except ImportError:  # The reference IPC worker imports this trusted module from its own root.
    from providers import Client

ROOT = Path(__file__).parent


class Module:
    def __init__(self, client_factory=None):
        self.client_factory = client_factory or Client
        self.client = None
        self.seen = OrderedDict()
        self.frames = self.sequence = 0

    def describe(self):
        return json.loads((ROOT / "manifest.json").read_text("utf-8"))

    def prepare(self, context, config):
        self.context = context
        self.client = self.client_factory(config)
        self.reset("prepare")

    def start(self, stream):
        self.stream = stream
        self.reset("start")

    def process(self, bundle):
        audio = []
        transferred = 0
        for packet in bundle.ports.get("text", ()):
            segment = TextSegment.from_packet(packet)
            key = (packet.stream_id, packet.epoch_id, segment.segment_id)
            if not segment.final or not segment.text.strip() or key in self.seen:
                continue
            samples = self.client.synthesize(segment.text, segment.language)
            transferred += len(samples)
            if transferred > 1048576:
                raise ValueError("tts_dispatch_audio_limit_exceeded")
            self.seen[key] = segment.revision
            if len(self.seen) > 256:
                self.seen.popitem(last=False)
            fmt = self.context.outputs["audio"]
            for offset in range(0, len(samples), 480):
                part = samples[offset : offset + 480]
                audio.append(
                    AudioPacket(
                        part,
                        fmt,
                        Fraction(self.frames, fmt.sample_rate),
                        self.stream.stream_id,
                        self.stream.epoch_id,
                        self.sequence,
                        frozenset({"generated"}),
                    )
                )
                self.frames += len(part)
                self.sequence += 1
        return PortBundle({"audio": tuple(audio), "text": bundle.ports.get("text", ())})

    def reset(self, reason):
        self.frames = self.sequence = 0
        self.seen.clear()

    def report_latency(self):
        return Latency(mode="variable")

    def flush(self):
        return FlushResult()

    def update_parameters(self, change):
        raise ValueError("TTS provider changes require reprepare")

    def stop(self):
        self.reset("stop")

    def dispose(self):
        self.client = None
        self.seen.clear()
