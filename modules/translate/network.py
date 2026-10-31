# SPDX-License-Identifier: Apache-2.0
"""One translation at a time in the process-isolated async graph worker."""

import json
from collections import OrderedDict
from pathlib import Path

from obliviousmic.sdk import (
    FlushResult,
    Latency,
    PortBundle,
    PublicModuleError,
    RequestStatus,
    StreamContext,
    TextSegment,
)

try:
    from .providers import Client
except ImportError:
    from providers import Client

ROOT = Path(__file__).parent


class Module:
    def __init__(self, client_factory=None):
        self.client_factory = client_factory or Client
        self.client = None
        self.seen = OrderedDict()
        self.requests = RequestStatus(PublicModuleError)

    def describe(self):
        return json.loads((ROOT / "obliviousmic.module.json").read_text("utf-8"))

    def prepare(self, context, config):
        self.context, self.config = context, dict(config)
        self.client = self.client_factory(config)  # Validation only; no paid warmup request.
        self.reset("prepare")

    def start(self, stream):
        self.reset("start")

    def process(self, bundle):
        output = []
        for packet in bundle.ports.get("text", ()):
            segment = TextSegment.from_packet(packet)
            key = (packet.stream_id, packet.epoch_id, segment.segment_id)
            previous = self.seen.get(key, (-1, False))
            if (
                not segment.text.strip()
                or (self.config["final_only"] and not segment.final)
                or segment.revision < previous[0]
                or (segment.revision == previous[0] and (previous[1] or not segment.final))
                or (previous[1] and not segment.final)
            ):
                continue
            self.seen[key] = (segment.revision, segment.final)
            self.seen.move_to_end(key)
            while len(self.seen) > 256:
                self.seen.popitem(last=False)
            translated = self.requests.call(self.client.translate, segment.text, segment.language)
            if translated is None:
                continue
            output.append(
                TextSegment(
                    segment.segment_id,
                    translated,
                    segment.start,
                    segment.end,
                    self.config["target_language"],
                    segment.final,
                    segment.revision,
                ).to_packet(
                    self.context.node_instance_id,
                    StreamContext(packet.stream_id, packet.epoch_id),
                    private=True,
                )
            )
        return PortBundle({"text": tuple(output)})

    def reset(self, reason):
        self.seen.clear()
        self.requests.reset()

    def report_monitor(self):
        return self.requests.report()

    def report_latency(self):
        return Latency(mode="variable")

    def flush(self):
        return FlushResult()

    def update_parameters(self, change):
        config = {**self.config, **change}
        client = self.client_factory(config)  # Engine/network initialization remains lazy.
        if hasattr(client, "detector") and hasattr(self.client, "detector"):
            client.detector = self.client.detector
        self.client, self.config = client, config
        # A settings change must not replay already translated segments.

    def stop(self):
        self.reset("stop")

    def dispose(self):
        self.client = None
        self.seen.clear()
