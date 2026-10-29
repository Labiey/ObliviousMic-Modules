# SPDX-License-Identifier: Apache-2.0
"""Own the isolated PCM worker and recover only its private Broadcast jobs."""

import json
import sys
import tempfile
from pathlib import Path

from obliviousmic.adapters.process import ProcessModule
from obliviousmic.sdk import PrepareContext

from .gateway import BroadcastError, Gateway

ROOT = Path(__file__).parent


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


def cleanup_jobs(directory, gateway_factory=Gateway):
    path = Path(directory) / "job.json"
    try:
        if path.stat().st_size > 4096:
            return
        job = json.loads(path.read_text("utf-8"))
        source = Path(job["sourcePath"]).resolve()
        if not source.is_relative_to(Path(directory).resolve()) or not isinstance(job["id"], str):
            return
        client = gateway_factory(job["port"], timeout=3)
        result = client.call(
            "get_file_processing_jobs", _timeout=3, _version=1, jobId=job["id"], limit=1
        )
        entries = [item for item in result.get("jobs", []) if item.get("id") == job["id"]]
        if any(
            item.get("status") in {"queued", "in_progress"}
            and Path(item.get("sourcePath", "")).resolve() == source
            for item in entries
        ):
            client.call(
                "control_file_processing_jobs",
                _timeout=3,
                _version=1,
                action="cancel",
                jobIds=[job["id"]],
            )
    except (OSError, ValueError, KeyError, TypeError):
        # Never fall back to queue-wide cancellation or a new submission.
        pass


class Module:
    MANIFEST = json.loads((ROOT / "obliviousmic.module.json").read_text("utf-8"))
    SCHEMA = json.loads((ROOT / "settings.schema.json").read_text("utf-8"))

    def __init__(self):
        self.worker = self.directory = None

    def describe(self):
        return self.MANIFEST

    def prepare(self, context, config):
        self.dispose()
        self.directory = tempfile.TemporaryDirectory(prefix="obmic-nvidia-session-")
        try:
            self.worker = ProcessModule(worker_command(), timeout=93)
            isolated = PrepareContext(
                context.node_instance_id, context.inputs, context.outputs, context.mode
            )
            self.worker.prepare(isolated, {**config, "_workdir": self.directory.name})
            self.worker.timeout = config["request_timeout_seconds"] + 3
        except BaseException:
            self.dispose()
            raise

    def start(self, stream):
        self.worker.start(stream)

    def process(self, bundle):
        return self.worker.process(bundle)

    def flush(self):
        return self.worker.flush()

    def report_latency(self):
        return self.worker.report_latency()

    def update_parameters(self, change):
        raise BroadcastError("broadcast_changes_require_reprepare")

    def reset(self, reason):
        self.worker.reset(reason)

    def stop(self):
        self.worker.stop()

    def dispose(self):
        try:
            if self.worker:
                self.worker.dispose()
                self.worker = None
        finally:
            if self.directory:
                cleanup_jobs(self.directory.name)
                self.directory.cleanup()
                self.directory = None
