# SPDX-License-Identifier: Apache-2.0
"""Private, bounded audio file jobs; output timing is explicit for the stream adapter."""

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .gateway import BroadcastError


@dataclass(frozen=True)
class FileResult:
    samples: np.ndarray
    sample_rate: int
    input_frames: int
    elapsed_seconds: float


class FileProcessor:
    def __init__(self, gateway, effects, *, timeout_seconds=45, workdir=None):
        if not 2 <= timeout_seconds <= 90 or not effects:
            raise BroadcastError("broadcast_processing_settings_invalid")
        self.gateway, self.effects = gateway, effects
        self.timeout_seconds = timeout_seconds
        self.workdir = workdir

    def process(self, samples, sample_rate=48000):
        samples = np.asarray(samples)
        if (
            samples.ndim != 2
            or samples.shape[1] != 1
            or sample_rate not in {16000, 24000, 32000, 44100, 48000}
            or not 0 < len(samples) <= sample_rate * 30
            or not np.isfinite(samples).all()
            or np.max(np.abs(samples), initial=0) > 1.0001
        ):
            raise BroadcastError("broadcast_input_audio_invalid")
        job = None
        terminal = False
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="obmic-nvidia-", dir=self.workdir) as folder:
            source, output = Path(folder) / "input.wav", Path(folder) / "output.wav"
            sf.write(source, samples, sample_rate, subtype="PCM_16")
            deadline = time.monotonic() + self.timeout_seconds
            try:
                job = self.gateway.call(
                    "submit_file_processing",
                    _timeout=min(self.timeout_seconds, 15),
                    _version=1,
                    inputPath=str(source.resolve()),
                    outputPath=str(output.resolve()),
                    effects=self.effects,
                )
                if not isinstance(job.get("id"), str) or not isinstance(job.get("batchId"), str):
                    raise BroadcastError("broadcast_job_response_invalid")
                if job.get("skippedEffects"):
                    raise BroadcastError("broadcast_requested_effect_skipped")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BroadcastError("broadcast_processing_timeout")
                update = self.gateway.call(
                    "wait_for_processing_update",
                    _timeout=min(remaining, 15),
                    _version=1,
                    batchId=job["batchId"],
                )
                while update.get("status") in {"queued", "in_progress"}:
                    remaining = deadline - time.monotonic()
                    if remaining < 1:
                        raise BroadcastError("broadcast_processing_timeout")
                    cursor = update.get("cursor")
                    if not isinstance(cursor, str):
                        raise BroadcastError("broadcast_job_response_invalid")
                    update = self.gateway.call(
                        "wait_for_processing_update",
                        _timeout=min(remaining, 15),
                        _version=1,
                        batchId=job["batchId"],
                        cursor=cursor,
                        timeoutMs=min(10000, int(remaining * 1000)),
                    )
                status = update.get("status")
                terminal = status in {"completed", "failed", "cancelled", "mixed"}
                if status == "paused_by_user":
                    raise BroadcastError("broadcast_paused_by_user")
                if status != "completed":
                    raise BroadcastError(
                        "broadcast_processing_"
                        + (
                            status
                            if status in {"failed", "cancelled", "mixed"}
                            else "invalid_status"
                        )
                    )
                jobs = update.get("jobs", [])
                completed = next((item for item in jobs if item.get("id") == job["id"]), None)
                if (
                    not completed
                    or completed.get("status") != "completed"
                    or completed.get("skippedEffects")
                ):
                    raise BroadcastError("broadcast_job_result_invalid")
                result_path = completed.get("outputPath")
                if (
                    not isinstance(result_path, str)
                    or output.is_symlink()
                    or output.resolve().parent != Path(folder).resolve()
                    or Path(result_path).resolve() != output.resolve()
                ):
                    raise BroadcastError("broadcast_output_path_mismatch")
                maximum_seconds = len(samples) / sample_rate + 1
                try:
                    with sf.SoundFile(output) as file:
                        if (
                            file.format not in {"WAV", "WAVEX"}
                            or file.channels != 1
                            or file.samplerate not in {16000, 24000, 32000, 44100, 48000}
                            or not 0 < file.frames <= file.samplerate * maximum_seconds
                        ):
                            raise BroadcastError("broadcast_output_audio_invalid")
                        result = file.read(dtype="float32", always_2d=True)
                        rate = file.samplerate
                except (OSError, RuntimeError):
                    raise BroadcastError("broadcast_output_audio_unavailable") from None
                if not np.isfinite(result).all() or np.max(np.abs(result), initial=0) > 1.0001:
                    raise BroadcastError("broadcast_output_audio_invalid")
                return FileResult(result, rate, len(samples), time.monotonic() - started)
            finally:
                if job and isinstance(job.get("id"), str) and not terminal:
                    # This exact owned job only: never all jobs, retry, resume or live setters.
                    try:
                        self.gateway.call(
                            "control_file_processing_jobs",
                            _timeout=3,
                            _version=1,
                            action="cancel",
                            jobIds=[job["id"]],
                        )
                    except (BroadcastError, OSError):
                        pass
