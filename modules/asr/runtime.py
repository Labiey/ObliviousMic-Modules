# SPDX-License-Identifier: Apache-2.0
"""Local utterance ASR; inference stays outside audio callbacks."""

import importlib
import json
import os
import shutil
import tempfile
from collections import deque
from fractions import Fraction
from pathlib import Path
from threading import RLock
from weakref import WeakSet

import numpy as np
from obliviousmic.sdk import FlushResult, Latency, PortBundle, TextSegment

ROOT = Path(__file__).parent


def configure_progress(progress_module=None):
    """Give only this backend's progress class a monitor-free implementation.

    Disabled upstream bars still create a global TMonitor. Do not stop an existing
    monitor or change tqdm's global settings used by downloads or embedding callers.
    """
    module = progress_module or importlib.import_module("faster_whisper.transcribe")
    original = module.tqdm
    if getattr(original, "_obmic_monitor_free", False):
        return

    class QuietProgress(original):
        monitor_interval = 0
        _instances = WeakSet()
        _obmic_monitor_free = True

    module.tqdm = QuietProgress


class WhisperSession:
    def __init__(self, files, config):
        from faster_whisper import WhisperModel

        configure_progress()
        self.directory = tempfile.TemporaryDirectory(prefix="obmic-whisper-")
        self.lock = RLock()
        try:
            for name, source in files.items():
                destination = Path(self.directory.name) / name
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copyfile(source, destination)
            self.model = WhisperModel(
                self.directory.name,
                device="cpu",
                compute_type="int8",
                cpu_threads=config["threads"],
                num_workers=1,
                local_files_only=True,
            )
        except Exception:
            self.directory.cleanup()
            raise

    def transcribe(self, samples, language):
        # faster-whisper's generator executes inference lazily: materialize under lock.
        with self.lock:
            segments, info = self.model.transcribe(
                samples,
                language=None if language == "auto" else language,
                beam_size=1,
                temperature=0,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            return tuple(segments), info.language


class Module:
    MANIFEST = json.loads((ROOT / "obliviousmic.module.json").read_text("utf-8"))
    SCHEMA = json.loads((ROOT / "settings.schema.json").read_text("utf-8"))

    def __init__(self, factory=None):
        self.factory = factory
        self.lease = self.processor = None
        self.buffers = deque()

    def describe(self):
        return self.MANIFEST

    def prepare(self, context, config):
        self.context, self.config = context, dict(config)
        if self.factory:
            self.processor = self.factory()
        else:
            resolve = context.capabilities["asset.read"]
            files = {
                filename: Path(resolve(config[key]))
                for filename, key in (
                    ("model.bin", "model_asset"),
                    ("config.json", "config_asset"),
                    ("tokenizer.json", "tokenizer_asset"),
                    ("vocabulary.txt", "vocabulary_asset"),
                )
            }
            self.lease = context.capabilities["resource.acquire"](
                files["model.bin"], config, lambda: WhisperSession(files, config)
            )
            self.processor = self.lease.value
        self.reset("prepare")

    def start(self, stream):
        self.stream = stream
        self.reset("start")

    def reset(self, reason):
        self.buffers.clear()
        self.frames = self.voice_frames = self.silent_frames = self.serial = 0
        self.origin = None

    def _clear_audio(self):
        self.buffers.clear()
        self.frames = self.voice_frames = self.silent_frames = 0
        self.origin = None

    def _recognize(self):
        outputs = []
        if self.voice_frames >= self.config["minimum_speech_ms"] * 16:
            samples = np.concatenate(tuple(self.buffers))
            segments, language = self.processor.transcribe(samples, self.config["language"])
            duration = Fraction(self.frames, 16000)
            for segment in segments:
                text = segment.text.strip()
                if not text:
                    continue
                start = max(Fraction(0), min(duration, Fraction(str(segment.start))))
                end = max(start, min(duration, Fraction(str(segment.end))))
                self.serial += 1
                outputs.append(
                    TextSegment(
                        f"utterance-{self.serial}",
                        text,
                        self.origin + start,
                        self.origin + end,
                        language,
                        final=True,
                    ).to_packet(self.context.node_instance_id, self.stream)
                )
        self._clear_audio()
        return outputs

    def process(self, bundle):
        outputs = []
        for packet in bundle.ports.get("audio", ()):
            if self.origin is None:
                self.origin = packet.media_start
            # Retained input must outlive the host's reusable capture buffer.
            self.buffers.append(packet.samples[:, 0].copy())
            self.frames += packet.frames
            energy = float(np.mean(packet.samples.astype(np.float64) ** 2))
            if energy >= 10 ** (self.config["silence_dbfs"] / 10):
                self.voice_frames += packet.frames
                self.silent_frames = 0
            else:
                self.silent_frames += packet.frames
            if self.voice_frames and (
                self.silent_frames >= self.config["silence_ms"] * 16
                or self.frames >= self.config["max_segment_seconds"] * 16000
            ):
                outputs.extend(self._recognize())
            elif not self.voice_frames:
                # Keep 200 ms pre-roll, never accumulate an unbounded silent room.
                while self.frames > 3200 and self.buffers:
                    drop = min(self.frames - 3200, len(self.buffers[0]))
                    first = self.buffers.popleft()
                    if drop < len(first):
                        self.buffers.appendleft(first[drop:])
                    self.frames -= drop
                    self.origin += Fraction(drop, 16000)
        return PortBundle({"text": tuple(outputs)})

    def flush(self):
        return FlushResult(PortBundle({"text": tuple(self._recognize())}))

    def update_parameters(self, change):
        self.config.update(change)

    def report_latency(self):
        return Latency(Fraction(self.config["max_segment_seconds"]), "variable")

    def stop(self):
        self._clear_audio()

    def dispose(self):
        self._clear_audio()
        self.processor = None
        if self.lease:
            self.lease.release()
            self.lease = None
