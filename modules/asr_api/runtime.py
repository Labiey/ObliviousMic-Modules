# SPDX-License-Identifier: Apache-2.0
"""Process-isolated remote ASR; credentials travel over IPC, never argv."""

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
    bootstrap = (
        "import sys;sys.path[:0]="
        + repr(paths)
        + ";from obliviousmic.adapters.worker import main;main()"
    )
    return [
        getattr(sys, "_base_executable", sys.executable),
        "-B",
        "-c",
        bootstrap,
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
        validate_config(config)
        try:
            self.worker = ProcessModule(
                worker_command(), timeout=config["request_timeout_seconds"] + 3
            )
            self.worker.prepare(
                PrepareContext(
                    context.node_instance_id, context.inputs, context.outputs, context.mode
                ),
                config,
            )
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

    def report_monitor(self):
        return self.worker._call("report_monitor")

    def update_parameters(self, change):
        raise ValueError("ASR API settings require reprepare")

    def reset(self, reason):
        self.worker.reset(reason)

    def stop(self):
        self.worker.stop()

    def dispose(self):
        if self.worker:
            self.worker.dispose()
            self.worker = None
