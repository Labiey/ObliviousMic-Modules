# SPDX-License-Identifier: Apache-2.0
"""Discover an existing portable GPT-SoVITS install; own only processes we start."""

import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .providers import NoRedirects, TTSError, local_url


class ExistingService:
    owned = False

    def close(self):
        pass


def probe(endpoint):
    url = local_url(endpoint) + "/openapi.json"
    try:
        opener = urllib.request.build_opener(NoRedirects(), urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=1) as response:
            raw = response.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                raise TTSError("tts_local_service_mismatch")
            data = json.loads(raw)
        if not {"/tts", "/set_gpt_weights", "/set_sovits_weights"} <= set(data.get("paths", {})):
            raise TTSError("tts_local_service_mismatch")
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
    except (ValueError, TypeError):
        raise TTSError("tts_local_service_mismatch") from None


def discover(configured=""):
    if configured:
        candidates = [Path(configured)]
    else:
        bases = (Path.home() / "Downloads", Path("D:/BaiduNetdiskDownload"))
        candidates = [
            p for base in bases if base.is_dir() for p in base.glob("GPT-SoVITS*") if p.is_dir()
        ]
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for directory in candidates[:10]:
        directory = directory.resolve()
        if all(
            (directory / p).is_file()
            for p in ("api_v2.py", "runtime/python.exe", "GPT_SoVITS/configs/tts_infer.yaml")
        ):
            return directory
    raise TTSError("tts_local_installation_not_found")


class OwnedService:
    owned = True

    def __init__(self, directory, endpoint, *, log_path=None, startup_seconds=120):
        self.process = None
        self.directory = tempfile.TemporaryDirectory(prefix="obmic-gpt-service-")
        self.closed = False
        self.created_at = None
        configuration = Path(self.directory.name) / "tts_infer.yaml"
        shutil.copyfile(directory / "GPT_SoVITS/configs/tts_infer.yaml", configuration)
        address = urllib.parse.urlparse(local_url(endpoint))
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        output = open(log_path, "wb") if log_path else subprocess.DEVNULL
        try:
            self.process = subprocess.Popen(
                [
                    str(directory / "runtime/python.exe"),
                    "-B",
                    str(directory / "api_v2.py"),
                    "-a",
                    "127.0.0.1" if address.hostname == "localhost" else address.hostname,
                    "-p",
                    str(address.port or 80),
                    "-c",
                    str(configuration),
                ],
                cwd=directory,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=output,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            import psutil

            try:
                self.created_at = psutil.Process(self.process.pid).create_time()
            except psutil.NoSuchProcess:
                raise TTSError("tts_local_start_failed") from None
            deadline = time.monotonic() + startup_seconds
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise TTSError("tts_local_start_failed")
                if probe(endpoint):
                    return
                time.sleep(0.25)
            raise TTSError("tts_local_start_timeout")
        except BaseException:
            self.close()
            raise
        finally:
            if log_path:
                output.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process and self.process.poll() is None:
            import psutil

            try:
                parent = psutil.Process(self.process.pid)
                children = (
                    parent.children(recursive=True)
                    if parent.create_time() == self.created_at
                    else []
                )
            except psutil.NoSuchProcess:
                children = []
            for child in reversed(children):
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            # Popen retains the original process handle on Windows (not just a PID).
            try:
                self.process.terminate()
            except OSError:
                if self.process.poll() is None:
                    raise
            try:
                self.process.wait(3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(3)
            _, alive = psutil.wait_procs(children, timeout=2)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
        self.directory.cleanup()


def acquire(config, acquire_service):
    endpoint = local_url(config["local_endpoint"])

    def start():
        if probe(endpoint):
            return ExistingService()
        if not config["auto_start_local"]:
            raise TTSError("tts_local_service_unavailable")
        parsed = urllib.parse.urlparse(endpoint)
        if parsed.scheme != "http":
            raise TTSError("tts_local_tls_service_requires_manual_start")
        try:
            with socket.create_connection((parsed.hostname, parsed.port or 80), timeout=0.3):
                raise TTSError("tts_local_endpoint_busy")
        except (ConnectionRefusedError, TimeoutError, OSError):
            pass
        return OwnedService(discover(config["local_install_directory"]), endpoint)

    return acquire_service("gpt-sovits:" + endpoint, start)
