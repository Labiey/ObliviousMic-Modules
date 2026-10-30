# SPDX-License-Identifier: Apache-2.0
"""Qwen-Audio-TTS task protocol. PCM stays in bounded memory inside the IPC worker."""

import json
import time
import uuid

import numpy as np
from websockets.exceptions import ConnectionClosed, InvalidStatus, WebSocketException
from websockets.sync.client import connect

try:
    from .providers import TTSError
except ImportError:
    from providers import TTSError


def synthesize(config, text, language, deadline, connector=None):
    region = config["qwen_region"]
    workspace = config.get("qwen_workspace_id", "")
    host = (
        workspace
        + (".cn-beijing" if region == "beijing" else ".ap-southeast-1")
        + ".maas.aliyuncs.com"
        if workspace
        else "dashscope.aliyuncs.com"
        if region == "beijing"
        else "dashscope-intl.aliyuncs.com"
    )
    task = uuid.uuid4().hex
    voice = config["qwen_voice"] or (
        "longanhuan_v3.1" if config["qwen_model"].startswith("qwen-audio-3.1-") else "longanhuan"
    )
    maximum = min(16 * 1024 * 1024, int(config["maximum_audio_seconds"] * 24000 * 2))
    pcm = bytearray()

    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise TTSError("tts_request_timeout")
        return value

    def send(ws, action, payload):
        remaining()
        ws.send(
            json.dumps(
                {
                    "header": {"action": action, "task_id": task, "streaming": "duplex"},
                    "payload": payload,
                },
                ensure_ascii=False,
            )
        )

    try:
        with (connector or connect)(
            "wss://" + host + "/api-ws/v1/inference",
            additional_headers={"Authorization": "Bearer " + config["qwen_api_key"]},
            open_timeout=remaining(),
            close_timeout=0.2,
            max_size=1024 * 1024,
            max_queue=4,
            compression=None,
            proxy=None,
            user_agent_header="ObliviousMic-TTS/1",
        ) as ws:
            send(
                ws,
                "run-task",
                {
                    "task_group": "audio",
                    "task": "tts",
                    "function": "SpeechSynthesizer",
                    "model": config["qwen_model"],
                    "input": {},
                    "parameters": {
                        "text_type": "PlainText",
                        "voice": voice,
                        "format": "pcm",
                        "sample_rate": 24000,
                        "volume": 50,
                        "rate": 1.0,
                        "pitch": 1.0,
                        "enable_ssml": False,
                        "language_hints": [language],
                    },
                },
            )
            started = False
            # Bound metadata floods as well as PCM bytes and wall-clock time.
            for _ in range(4096):
                event = ws.recv(timeout=remaining())
                remaining()
                if isinstance(event, bytes):
                    if not started:
                        raise TTSError("tts_qwen_stream_invalid")
                    if len(pcm) + len(event) > maximum:
                        raise TTSError("tts_response_exceeds_limit")
                    pcm.extend(event)
                    continue
                try:
                    message = json.loads(event)
                    header = message["header"]
                    if not isinstance(header, dict) or header.get("task_id") != task:
                        raise ValueError()
                    kind = header["event"]
                except (ValueError, TypeError, KeyError):
                    raise TTSError("tts_qwen_stream_invalid") from None
                if kind == "task-failed":
                    if not isinstance(header.get("error_code"), str):
                        raise TTSError("tts_qwen_stream_invalid")
                    code = {
                        "InvalidParameter": "tts_qwen_parameters_invalid",
                        "InvalidApiKey": "tts_provider_http_401",
                        "Throttling": "tts_provider_http_429",
                        "Throttling.RateQuota": "tts_provider_http_429",
                    }.get(header.get("error_code"), "tts_qwen_task_failed")
                    raise TTSError(code)
                if kind == "task-started" and not started:
                    started = True
                    send(ws, "continue-task", {"input": {"text": text}})
                    send(ws, "finish-task", {"input": {}})
                elif kind == "task-finished" and started:
                    if not pcm or len(pcm) % 2:
                        raise TTSError("tts_audio_response_invalid")
                    return np.frombuffer(pcm, dtype="<i2").astype(np.float32).reshape(-1, 1) / 32768
                elif kind != "result-generated" or not started:
                    raise TTSError("tts_qwen_stream_invalid")
            raise TTSError("tts_response_exceeds_limit")
    except TTSError:
        raise
    except InvalidStatus as error:
        raise TTSError("tts_provider_http_" + str(error.response.status_code)) from None
    except TimeoutError:
        raise TTSError("tts_request_timeout") from None
    except (ConnectionClosed, WebSocketException, OSError):
        raise TTSError("tts_provider_unavailable_or_timeout") from None
