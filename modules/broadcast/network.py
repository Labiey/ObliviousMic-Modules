# SPDX-License-Identifier: Apache-2.0
"""Buffered arbitrary PCM enhancement with context, content alignment and exact EOS."""

import json
from collections import deque
from fractions import Fraction
from pathlib import Path

import numpy as np
import soxr
from obliviousmic.sdk import AudioPacket, FlushResult, Latency, PortBundle

try:
    from .files import FileProcessor
    from .gateway import BroadcastError, Gateway, audio_effects
except ImportError:
    from files import FileProcessor
    from gateway import BroadcastError, Gateway, audio_effects

ROOT = Path(__file__).parent
RATE = 48000
MAX_FRAMES = 1048576


class Module:
    def __init__(self, gateway_factory=Gateway, processor_factory=FileProcessor):
        self.gateway_factory, self.processor_factory = gateway_factory, processor_factory
        self.processor = None

    def describe(self):
        return json.loads((ROOT / "obliviousmic.module.json").read_text("utf-8"))

    def prepare(self, context, config):
        self.context = context
        if (
            context.inputs["audio"].sample_rate != RATE
            or context.inputs["audio"].channels != 1
            or context.outputs["audio"] != context.inputs["audio"]
        ):
            raise BroadcastError("broadcast_requires_48k_mono")
        port = config["gateway_port"] or None
        gateway = self.gateway_factory(port)
        effects = audio_effects(
            gateway.call("describe_effects", section="microphone"),
            config["algorithm"],
            config["strength"],
            config["mic_profile"],
        )
        window = config["window_seconds"] or (8 if config["algorithm"] == "studio_voice" else 2)
        if not 2 <= window <= 12 or (
            context.mode == "realtime" and config["algorithm"] == "studio_voice" and window < 8
        ):
            raise BroadcastError("broadcast_buffer_window_too_short")
        delay = config["alignment_delay_ms"]
        if delay == -1:
            if not gateway.version.startswith("2.3."):
                raise BroadcastError("broadcast_version_needs_explicit_alignment_delay")
            delay = 40 if config["algorithm"] == "studio_voice" else 70
        if not 0 <= delay <= 200:
            raise BroadcastError("broadcast_alignment_delay_invalid")
        self.delay = round(delay * RATE / 1000)
        self.window = round(window * RATE)
        self.lookahead = max(RATE // 10, self.delay + 960)
        self.context_frames = RATE // 4
        workdir = config.get("_workdir")

        def journal(job):
            if not workdir:
                return
            path = Path(workdir) / "job.json"
            if job is None:
                path.unlink(missing_ok=True)
            else:
                staging = path.with_suffix(".new")
                staging.write_text(json.dumps({**job, "port": gateway.port}), encoding="utf-8")
                staging.replace(path)

        self.processor = self.processor_factory(
            gateway,
            effects,
            timeout_seconds=config["request_timeout_seconds"],
            workdir=workdir,
            job_state=journal,
        )
        # Cold effect/model loading happens before the first live utterance.
        self.processor.process(np.zeros((RATE // 2, 1), np.float32), RATE)
        self._clear()

    def _clear(self):
        self.parts = deque()
        self.pending = 0
        self.history = np.zeros((self.context_frames, 1), np.float32)
        self.origin = self.input_end = None
        self.frames = self.sequence = 0
        self.flags = set()

    def start(self, stream):
        self.stream = stream
        self._clear()

    def _peek(self, count):
        chunks = []
        for part in self.parts:
            take = min(len(part), count)
            chunks.append(part[:take])
            count -= take
            if count == 0:
                break
        if count:
            raise BroadcastError("broadcast_buffer_underflow")
        return np.concatenate(chunks, axis=0)

    def _consume(self, count):
        self.pending -= count
        while count:
            part = self.parts.popleft()
            if len(part) > count:
                self.parts.appendleft(part[count:])
                return
            count -= len(part)

    def _enhance(self, count):
        raw = self._peek(min(self.pending, count + self.lookahead))
        missing = count + self.lookahead - len(raw)
        batch = np.concatenate((self.history, raw, np.zeros((missing, 1), np.float32)), axis=0)
        result = self.processor.process(batch, RATE)
        samples = result.samples
        if result.sample_rate != RATE:
            samples = soxr.resample(samples, result.sample_rate, RATE, quality="HQ")
        offset = self.context_frames + self.delay
        if len(samples) < offset + count:
            raise BroadcastError("broadcast_output_did_not_drain_context")
        samples = np.ascontiguousarray(samples[offset : offset + count], dtype=np.float32)
        self.history = np.concatenate((self.history, raw[:count]), axis=0)[
            -self.context_frames :
        ].copy()
        self._consume(count)
        output = []
        for offset in range(0, count, 960):
            part = samples[offset : offset + 960]
            flags = self.flags if self.sequence == 0 else self.flags - {"discontinuity"}
            output.append(
                AudioPacket(
                    part,
                    self.context.outputs["audio"],
                    self.origin + Fraction(self.frames, RATE),
                    self.stream.stream_id,
                    self.stream.epoch_id,
                    self.sequence,
                    frozenset(flags),
                )
            )
            self.frames += len(part)
            self.sequence += 1
        return output

    def process(self, bundle):
        for packet in bundle.ports.get("audio", ()):
            if self.input_end is not None and packet.media_start != self.input_end:
                raise BroadcastError("broadcast_discontinuous_input_requires_reset")
            if (packet.stream_id, packet.epoch_id) != (self.stream.stream_id, self.stream.epoch_id):
                raise BroadcastError("broadcast_stream_mismatch")
            if self.pending + packet.frames > MAX_FRAMES:
                raise BroadcastError("broadcast_input_dispatch_exceeds_limit")
            if self.origin is None:
                self.origin = packet.media_start
            self.parts.append(np.array(packet.samples, dtype=np.float32, copy=True))
            self.pending += packet.frames
            self.input_end = packet.media_end
            self.flags.update(packet.flags)
        output = []
        while self.pending >= self.window + self.lookahead:
            output.extend(self._enhance(self.window))
        return PortBundle({"audio": tuple(output)})

    def flush(self):
        output = self._enhance(self.pending) if self.pending else []
        return FlushResult(PortBundle({"audio": tuple(output)}))

    def report_latency(self):
        return Latency(Fraction(self.window + self.lookahead, RATE), "variable")

    def reset(self, reason):
        self._clear()

    def update_parameters(self, change):
        raise BroadcastError("broadcast_changes_require_reprepare")

    def stop(self):
        self._clear()

    def dispose(self):
        if self.processor:
            self._clear()
            self.processor = None
