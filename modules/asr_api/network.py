# SPDX-License-Identifier: Apache-2.0
"""Bounded utterance segmentation in a cancellable network worker."""

import json
from collections import deque
from fractions import Fraction
from pathlib import Path

import numpy as np
from obliviousmic.sdk import FlushResult, Latency, PortBundle, StreamContext, TextSegment

try:
    from .providers import RATE, ASRError, Client
except ImportError:
    from providers import RATE, ASRError, Client

ROOT = Path(__file__).parent


class Module:
    def __init__(self, client_factory=None):
        self.client_factory = client_factory or Client
        self.client = None
        self.buffers = deque()

    def describe(self):
        return json.loads((ROOT / "obliviousmic.module.json").read_text("utf-8"))

    def prepare(self, context, config):
        self.context, self.config = context, dict(config)
        self.client = self.client_factory(config)  # No network/warmup/billable request.
        self.reset("prepare")

    def start(self, stream):
        self.reset("start")

    def _clear(self):
        self.buffers.clear()
        self.frames = self.voice_frames = self.silent_frames = 0
        self.origin = None

    def reset(self, reason):
        self._clear()
        self.expected_end = self.identity = None
        self.serial = 0
        self.finished = False

    def _recognize(self):
        result = []
        try:
            if self.voice_frames >= self.config["minimum_speech_ms"] * RATE // 1000:
                text, language = self.client.transcribe(np.concatenate(tuple(self.buffers)))
                if text:
                    self.serial += 1
                    result.append(
                        TextSegment(
                            f"api-utterance-{self.serial}",
                            text,
                            self.origin,
                            self.origin + Fraction(self.frames, RATE),
                            language,
                        ).to_packet(self.context.node_instance_id, StreamContext(*self.identity))
                    )
        finally:
            self._clear()
        return result

    def process(self, bundle):
        if self.finished:
            raise ASRError("asr_api_stream_drained")
        result = []
        for packet in bundle.ports.get("audio", ()):
            if packet.format.sample_rate != RATE or packet.format.channels != 1:
                raise ASRError("asr_api_audio_invalid")
            if not packet.frames:
                continue
            if not np.isfinite(packet.samples).all():
                raise ASRError("asr_api_audio_invalid")
            identity = (packet.stream_id, packet.epoch_id)
            if (
                self.identity != identity
                or self.expected_end != packet.media_start
                or "discontinuity" in packet.flags
            ):
                self._clear()
            self.identity, self.expected_end = identity, packet.media_end
            # Iterate at <=20 ms even if called directly without a host chunk adapter.
            offset = 0
            while offset < packet.frames:
                count = min(
                    320,
                    packet.frames - offset,
                    self.config["max_segment_seconds"] * RATE - self.frames,
                )
                samples = packet.samples[offset : offset + count, 0]
                if self.origin is None:
                    self.origin = packet.media_start + Fraction(offset, RATE)
                self.buffers.append(samples.copy())
                self.frames += len(samples)
                energy = float(np.mean(samples.astype(np.float64) ** 2))
                if energy >= 10 ** (self.config["silence_dbfs"] / 10):
                    self.voice_frames += len(samples)
                    self.silent_frames = 0
                else:
                    self.silent_frames += len(samples)
                if self.voice_frames and (
                    self.frames >= self.config["max_segment_seconds"] * RATE
                    or self.silent_frames >= self.config["silence_ms"] * RATE // 1000
                ):
                    result.extend(self._recognize())
                elif not self.voice_frames:
                    while self.frames > 3200 and self.buffers:
                        first = self.buffers.popleft()
                        drop = min(len(first), self.frames - 3200)
                        if drop < len(first):
                            self.buffers.appendleft(first[drop:])
                        self.frames -= drop
                        self.origin += Fraction(drop, RATE)
                offset += count
        return PortBundle({"text": tuple(result)})

    def flush(self):
        if self.finished:
            return FlushResult()
        self.finished = True
        return FlushResult(PortBundle({"text": tuple(self._recognize())}))

    def report_latency(self):
        return Latency(Fraction(self.config["max_segment_seconds"]), "variable")

    def update_parameters(self, change):
        raise ValueError("ASR API settings require reprepare")

    def stop(self):
        self.reset("stop")

    def dispose(self):
        self.client = None
        self.reset("dispose")
