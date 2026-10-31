# SPDX-License-Identifier: Apache-2.0
"""Bounded cloud synthesis worker ownership; no local model or voice assets."""

import json
import sys
from pathlib import Path

from obliviousmic.adapters.process import ProcessModule
from obliviousmic.sdk import PrepareContext

from .providers import validate_config

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


class Module:
    MANIFEST = json.loads((ROOT / "obliviousmic.module.json").read_text("utf-8"))
    SCHEMA = json.loads((ROOT / "settings.schema.json").read_text("utf-8"))

    def __init__(self):
        self.worker = None

    def describe(self):
        return self.MANIFEST

    def prepare(self, context, config):
        self.dispose()
        config = dict(config)
        validate_config(config)
        self.config = config
        try:
            self.worker = ProcessModule(
                worker_command(),
                timeout=self.SCHEMA["properties"]["request_timeout_seconds"]["maximum"] + 3,
            )
            isolated = PrepareContext(
                context.node_instance_id, context.inputs, context.outputs, context.mode
            )
            # Credentials travel only over bounded IPC, never command arguments.
            self.worker.prepare(isolated, config)
        except BaseException:
            self.dispose()
            raise

    def start(self, stream):
        self.worker.start(stream)

    def process(self, bundle):
        return self.worker.process(bundle)

    def update_parameters(self, change):
        config = {**self.config, **change}
        self.validate_parameters(config)
        self.worker.update_parameters(change)
        self.config = config

    def validate_parameters(self, config):
        validate_config(config)

    def report_latency(self):
        return self.worker.report_latency()

    def report_monitor(self):
        return self.worker._call("report_monitor")

    def flush(self):
        return self.worker.flush()

    def reset(self, reason):
        self.worker.reset(reason)

    def stop(self):
        self.worker.stop()

    def dispose(self):
        if self.worker:
            self.worker.dispose()
            self.worker = None
