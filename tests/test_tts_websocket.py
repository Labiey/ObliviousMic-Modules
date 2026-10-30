import json
import time

import numpy as np
import pytest
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response

from modules.tts.providers import Client, TTSError
from modules.tts.qwen_websocket import synthesize
from tests.test_tts import config


class Socket:
    def __init__(self, events):
        self.events = iter(events)
        self.sent = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True

    def send(self, message):
        self.sent.append(json.loads(message))

    def recv(self, timeout):
        assert timeout > 0
        event = next(self.events)
        if isinstance(event, Exception):
            raise event
        if isinstance(event, bytes):
            return event
        header = {"task_id": self.sent[0]["header"]["task_id"], "event": event}
        if isinstance(event, dict):
            header.update(event)
        return json.dumps({"header": header})


def client(socket, **change):
    calls = []

    def connector(uri, **kwargs):
        calls.append((uri, kwargs))
        return socket

    def no_http(*_args, **_kwargs):
        raise AssertionError("WebSocket models must not send an HTTP synthesis request")

    return Client(
        config(qwen_model="qwen-audio-3.1-tts-flash", **change), no_http, connector
    ), calls


@pytest.mark.parametrize(
    "region,workspace,host",
    [
        ("beijing", "", "dashscope.aliyuncs.com"),
        ("singapore", "", "dashscope-intl.aliyuncs.com"),
        ("beijing", "space-123", "space-123.cn-beijing.maas.aliyuncs.com"),
        ("singapore", "space-123", "space-123.ap-southeast-1.maas.aliyuncs.com"),
    ],
)
def test_stream_task_order_pcm_fragments_and_credentials(region, workspace, host):
    pcm = np.array([0, -32768, 32767], dtype="<i2").tobytes()
    socket = Socket(["task-started", "result-generated", pcm[:3], pcm[3:], "task-finished"])
    instance, calls = client(socket, qwen_region=region, qwen_workspace_id=workspace)
    output = instance.synthesize("private fixture", "ja")
    assert output.dtype == np.float32 and output.shape == (3, 1)
    np.testing.assert_allclose(output[:, 0], [0, -1, 32767 / 32768])
    assert calls[0][0] == "wss://" + host + "/api-ws/v1/inference"
    assert calls[0][1]["additional_headers"] == {"Authorization": "Bearer fixture-key"}
    assert calls[0][1]["proxy"] is None and calls[0][1]["max_queue"] == 4
    assert [v["header"]["action"] for v in socket.sent] == [
        "run-task",
        "continue-task",
        "finish-task",
    ]
    assert len({v["header"]["task_id"] for v in socket.sent}) == 1
    assert socket.sent[0]["payload"]["parameters"]["voice"] == "longanhuan_v3.1"
    assert socket.sent[0]["payload"]["parameters"]["language_hints"] == ["ja"]
    assert socket.sent[1]["payload"]["input"]["text"] == "private fixture"
    assert "fixture-key" not in json.dumps(socket.sent) and socket.closed


def test_custom_voice_is_not_replaced_and_tasks_have_distinct_ids():
    sockets = [Socket(["task-started", b"\x00\x01", "task-finished"]) for _ in range(2)]
    for socket in sockets:
        instance, _ = client(socket, qwen_voice="custom-private-voice")
        instance.synthesize("fixture")
        assert socket.sent[0]["payload"]["parameters"]["voice"] == "custom-private-voice"
    assert sockets[0].sent[0]["header"]["task_id"] != sockets[1].sent[0]["header"]["task_id"]


@pytest.mark.parametrize(
    "events,code",
    [
        ([b"\x00\x01"], "tts_qwen_stream_invalid"),
        (["task-finished"], "tts_qwen_stream_invalid"),
        (["task-started", "task-started"], "tts_qwen_stream_invalid"),
        (["task-started", "task-finished"], "tts_audio_response_invalid"),
        (["task-started", b"\x00", "task-finished"], "tts_audio_response_invalid"),
        (
            ["task-started", {"event": "result-generated", "task_id": "stale"}],
            "tts_qwen_stream_invalid",
        ),
        ([TimeoutError("private timeout")], "tts_request_timeout"),
        ([OSError("private key")], "tts_provider_unavailable_or_timeout"),
        (
            [
                {
                    "event": "task-failed",
                    "error_code": "InvalidParameter",
                    "error_message": "private key and prompt",
                }
            ],
            "tts_qwen_parameters_invalid",
        ),
        ([{"event": "task-failed", "error_code": "unknown private key"}], "tts_qwen_task_failed"),
        ([{"event": "task-failed", "error_code": []}], "tts_qwen_stream_invalid"),
    ],
)
def test_invalid_or_failed_stream_is_closed_and_never_returns_partial_audio(events, code):
    socket = Socket(events)
    instance, _ = client(socket)
    with pytest.raises(TTSError) as error:
        instance.synthesize("private fixture")
    assert str(error.value) == code and socket.closed


def test_duration_and_metadata_floods_are_bounded():
    for events in (
        ["task-started", b"\x00" * 48002],
        ["task-started"] + ["result-generated"] * 4096,
    ):
        socket = Socket(events)
        instance, _ = client(socket, maximum_audio_seconds=1)
        with pytest.raises(TTSError, match="tts_response_exceeds_limit"):
            instance.synthesize("fixture")
        assert socket.closed
    with pytest.raises(TTSError, match="tts_request_timeout"):
        synthesize(config(), "fixture", "en", time.monotonic() - 1)


@pytest.mark.parametrize(
    "workspace", ["evil.example", "space/path", "space?key=fixture", "a@evil", "UPPER"]
)
def test_workspace_cannot_change_authorization_destination(workspace):
    with pytest.raises(TTSError, match="tts_qwen_workspace_invalid"):
        Client(config(qwen_workspace_id=workspace))


@pytest.mark.parametrize("status", [302, 401, 403])
def test_handshake_errors_are_public_and_do_not_retry_or_forward_credentials(status):
    calls = []

    def rejected(uri, **kwargs):
        calls.append(uri)
        raise InvalidStatus(Response(status, "private prompt", Headers()))

    instance = Client(config(qwen_model="qwen-audio-3.1-tts-flash"), websocket_connect=rejected)
    with pytest.raises(TTSError) as error:
        instance.synthesize("private fixture")
    assert str(error.value) == "tts_provider_http_" + str(status)
    assert len(calls) == 1
