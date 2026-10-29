# SPDX-License-Identifier: Apache-2.0
"""Nonblocking bounded transport for an owned Windows subtitle window."""

import json
import os
import queue
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from threading import Event, Thread
from time import monotonic

from obliviousmic.sdk import FlushResult, Latency, TextSegment
from obliviousmic.text_hotkey import parse_hotkey

ROOT = Path(__file__).parent


class OverlaySession:
    def __init__(self):
        if sys.platform != "win32":
            raise ValueError("subtitle_window_requires_windows")
        self.updates = queue.Queue(maxsize=64)
        self.ready = Event()
        self.reader_done = Event()
        self.status = {"state": "starting"}
        self.closed = False
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        bootstrap = (
            "import sys,runpy;sys.path[:0]="
            + repr(sys.path)
            + ";runpy.run_path("
            + repr(str(ROOT / "window.py"))
            + ",run_name='__main__')"
        )
        self.process = subprocess.Popen(
            [getattr(sys, "_base_executable", sys.executable), "-B", "-c", bootstrap],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
            env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self.writer = Thread(target=self._write, name="subtitle-writer", daemon=True)
        self.reader = Thread(target=self._read, name="subtitle-status", daemon=True)
        self.writer.start()
        self.reader.start()
        if not self.ready.wait(5) or self.status.get("state") != "ready":
            self.close()
            raise ValueError("subtitle_window_unavailable_or_tkinter_missing")

    def _write(self):
        try:
            while True:
                value = self.updates.get()
                if value is None:
                    break
                self.process.stdin.write(json.dumps(value, ensure_ascii=True) + "\n")
                self.process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass
        finally:
            try:
                self.process.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    status = json.loads(line)
                except ValueError:
                    continue
                if status.get("state") in {
                    "ready",
                    "displayed",
                    "visible",
                    "hidden",
                    "closed",
                    "error",
                }:
                    self.status = status
                    self.ready.set()
        finally:
            self.process.stdout.close()
            self.reader_done.set()
            self.ready.set()

    def submit(self, value):
        if self.closed:
            raise ValueError("subtitle_window_closed")
        if self.process.poll() is not None:
            self.reader_done.wait(0.1)
            if self.status.get("state") == "closed" and self.status.get("user"):
                return False
            raise ValueError("subtitle_window_exited")
        while True:
            try:
                self.updates.put_nowait(value)
                return True
            except queue.Full:
                try:
                    self.updates.get_nowait()
                except queue.Empty:
                    pass

    def close(self):
        if self.closed:
            return
        self.closed = True
        while True:
            try:
                self.updates.get_nowait()
            except queue.Empty:
                break
        self.updates.put_nowait(None)
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        self.writer.join(1)
        self.reader.join(1)


class Module:
    MANIFEST = json.loads((ROOT / "obliviousmic.module.json").read_text("utf-8"))
    SCHEMA = json.loads((ROOT / "settings.schema.json").read_text("utf-8"))

    def __init__(self, factory=None):
        self.factory = factory or OverlaySession
        self.session = None
        self.generation = 0
        self.user_closed = False
        self.current = None
        self.current_at = 0
        self.seen = OrderedDict()

    def describe(self):
        return self.MANIFEST

    def prepare(self, context, config):
        self.context, self.config = context, dict(config)
        parse_hotkey(self.config.get("overlay_hotkey", ""))
        self.reset("prepare")

    def start(self, stream):
        self.reset("start")
        if self.session is None and self._active() and self.config.get("overlay_hotkey"):
            self.session = self.factory()
            self.session.submit(self._payload())

    def _active(self):
        return self.config["enabled"] and (
            self.context.mode == "realtime" or self.config["show_in_offline"]
        )

    def _payload(self, segment=None):
        return {
            "generation": self.generation,
            "config": dict(self.config),
            "text": segment.text if segment else "",
            "id": segment.segment_id if segment else "",
            "revision": segment.revision if segment else 0,
        }

    def reset(self, reason):
        self.generation += 1
        self.seen.clear()
        self.current = None
        self.user_closed = False
        if self.session:
            self.session.submit(self._payload())

    def process(self, bundle):
        active = self._active()
        for packet in bundle.ports.get("text", ()):
            text = TextSegment.from_packet(packet)
            key = (packet.stream_id, packet.epoch_id, text.segment_id)
            previous = self.seen.get(key, -1)
            if text.revision <= previous:
                continue
            self.seen[key] = text.revision
            self.seen.move_to_end(key)
            if len(self.seen) > 256:
                self.seen.popitem(last=False)
            self.current, self.current_at = text, monotonic()
            if active and not self.user_closed:
                if self.session is None:
                    self.session = self.factory()
                if not self.session.submit(self._payload(text)):
                    self.user_closed = True
                    self.session.close()
                    self.session = None
        return bundle

    def update_parameters(self, change):
        was_enabled = self.config["enabled"]
        self.config.update(change)
        if not self.config["enabled"]:
            self.stop()
        elif not was_enabled:
            self.user_closed = False
            self.seen.clear()
        elif self.session:
            current = (
                self.current
                if monotonic() - self.current_at < self.config["display_seconds"]
                else None
            )
            self.session.submit(self._payload(current))

    def report_latency(self):
        return Latency()

    def flush(self):
        return FlushResult()

    def stop(self):
        if self.session:
            self.session.close()
            self.session = None

    def dispose(self):
        self.stop()
        self.seen.clear()
