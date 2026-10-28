import base64
import io
import json
import urllib.error
import wave
from contextlib import contextmanager
from email import policy
from email.parser import BytesParser
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from time import monotonic, sleep

import numpy as np
import pytest
from obliviousmic.sdk import (
    AudioFormat,
    AudioPacket,
    PortBundle,
    PrepareContext,
    StreamContext,
    TextSegment,
)

from modules.asr_api.network import Module as NetworkModule
from modules.asr_api.providers import MAX_RESPONSE_BYTES, ASRError, Client, endpoint
from modules.asr_api.runtime import Module


def config(**changes):
    return {
        **{k: v["default"] for k, v in Module.SCHEMA["properties"].items()},
        "api_key": "fixture-key",
        "base_url": "https://api.example/v1",
        **changes,
    }


def parsed_form(request):
    message = BytesParser(policy=policy.default).parsebytes(
        ("Content-Type: " + request.get_header("Content-type") + "\r\n\r\n").encode() + request.data
    )
    return {
        part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
        for part in message.iter_parts()
    }


def test_openai_compatible_wav_protocol_and_optional_local_auth():
    calls = []

    def open(request, **kwargs):
        calls.append((request, kwargs))
        return io.BytesIO(json.dumps({"text": "  你好 日本語 hello  ", "language": "ja"}).encode())

    client = Client(config(language="zh"), opener=open)
    assert client.transcribe(np.full(1601, 0.1, np.float32)) == ("你好 日本語 hello", "ja")
    request, options = calls[0]
    assert request.full_url == "https://api.example/v1/audio/transcriptions"
    assert options["timeout"] == 15 and request.get_header("Authorization") == "Bearer fixture-key"
    parts = parsed_form(request)
    assert (
        parts["model"] == b"whisper-1"
        and parts["language"] == b"zh"
        and parts["response_format"] == b"json"
    )
    with wave.open(io.BytesIO(parts["file"]), "rb") as audio:
        assert (
            audio.getframerate(),
            audio.getnchannels(),
            audio.getsampwidth(),
            audio.getnframes(),
        ) == (16000, 1, 2, 1601)
    calls.clear()
    client = Client(
        config(base_url="http://127.0.0.1:9000/v1/audio/transcriptions", api_key=""), opener=open
    )
    client.transcribe(np.zeros(1600, np.float32))
    assert calls[0][0].full_url.endswith("/v1/audio/transcriptions")
    assert calls[0][0].get_header("Authorization") is None
    assert "language" not in parsed_form(calls[0][0])


@pytest.mark.parametrize("status", ["Success", "NoMatch", "InitialSilenceTimeout", "BabbleTimeout"])
def test_azure_speech_protocol_and_empty_speech_results(status):
    calls = []

    def open(request, **kwargs):
        calls.append(request)
        return io.BytesIO(
            json.dumps({"RecognitionStatus": status, "DisplayText": "こんにちは。"}).encode()
        )

    client = Client(
        config(provider="azure", azure_api_key="azure-fixture", azure_language="ja-JP"), opener=open
    )
    text, lang = client.transcribe(np.zeros(4000, np.float32))
    assert text == ("こんにちは。" if status == "Success" else "") and lang == "ja"
    request = calls[0]
    assert (
        request.full_url
        == "https://eastasia.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language=ja-JP&format=simple"
    )
    assert request.get_header("Ocp-apim-subscription-key") == "azure-fixture"
    assert request.get_header("Authorization") is None
    with wave.open(io.BytesIO(request.data), "rb") as audio:
        assert audio.getnframes() == 4000


@pytest.mark.parametrize(
    "region,host",
    [("beijing", "dashscope.aliyuncs.com"), ("singapore", "dashscope-intl.aliyuncs.com")],
)
@pytest.mark.parametrize("language", ["auto", "zh", "en", "ja"])
def test_bailian_audio_message_contract_and_language_options(region, host, language):
    calls = []

    def open(request, **kwargs):
        calls.append(request)
        return io.BytesIO(
            json.dumps({"choices": [{"message": {"content": "  你好 hello 日本語  "}}]}).encode()
        )

    result = Client(
        config(
            provider="bailian",
            bailian_api_key="bailian-fixture",
            bailian_region=region,
            language=language,
            bailian_enable_itn=True,
        ),
        opener=open,
    ).transcribe(np.full(1601, 0.1, np.float32))
    assert result == ("你好 hello 日本語", language)
    assert len(calls) == 1
    request = calls[0]
    assert request.full_url == f"https://{host}/compatible-mode/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer bailian-fixture"
    body = json.loads(request.data)
    assert body["model"] == "qwen3-asr-flash" and body["stream"] is False
    assert body["asr_options"] == {
        "enable_itn": True,
        **({"language": language} if language != "auto" else {}),
    }
    message = body["messages"][0]
    assert len(body["messages"]) == 1 and message["role"] == "user"
    content = message["content"][0]
    assert content["type"] == "input_audio"
    prefix, encoded = content["input_audio"]["data"].split(",", 1)
    assert prefix == "data:audio/wav;base64"
    with wave.open(io.BytesIO(base64.b64decode(encoded, validate=True)), "rb") as audio:
        assert (
            audio.getframerate(),
            audio.getnchannels(),
            audio.getsampwidth(),
            audio.getnframes(),
        ) == (16000, 1, 2, 1601)


@pytest.mark.parametrize("suffix", ["", "/chat/completions"])
def test_bailian_workspace_endpoint(suffix):
    calls = []

    def open(request, **kwargs):
        calls.append(request)
        return io.BytesIO(b'{"choices":[{"message":{"content":""}}]}')

    base = "https://workspace-example.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    result = Client(
        config(provider="bailian", bailian_api_key="fixture", bailian_base_url=base + suffix),
        opener=open,
    ).transcribe(np.zeros(1600, np.float32))
    assert result == ("", "auto")
    assert calls[0].full_url == base + "/chat/completions"


@pytest.mark.parametrize(
    "change,code",
    [
        ({"bailian_api_key": ""}, "key_required"),
        ({"bailian_api_key": "fixture\nsecret"}, "key_invalid"),
        ({"bailian_model": ""}, "model_required"),
        ({"bailian_region": "invalid"}, "bailian_region_invalid"),
        ({"bailian_base_url": "http://127.0.0.1/compatible-mode/v1"}, "bailian_endpoint_invalid"),
        (
            {"bailian_base_url": "https://dashscope.aliyuncs.com.evil.invalid/compatible-mode/v1"},
            "bailian_endpoint_invalid",
        ),
        (
            {"bailian_base_url": "https://dashscope.aliyuncs.com:8443/compatible-mode/v1"},
            "bailian_endpoint_invalid",
        ),
        (
            {
                "bailian_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/audio/transcriptions"
            },
            "bailian_endpoint_invalid",
        ),
    ],
)
def test_bailian_invalid_config_never_sends_audio(change, code):
    with pytest.raises(ASRError, match="asr_api_" + code):
        Client(
            config(provider="bailian", **{"bailian_api_key": "fixture", **change}),
            opener=lambda *_a, **_k: pytest.fail("unexpected network"),
        )


@pytest.mark.parametrize(
    "result",
    [
        {"code": "InvalidApiKey", "message": "PRIVATE-KEY"},
        {"choices": []},
        {"choices": [{"message": {"content": None}}]},
        {"choices": [{"message": {"content": [{"text": "wrong-protocol"}]}}]},
    ],
)
def test_bailian_rejects_malformed_or_error_responses(result):
    with pytest.raises(ASRError) as exc:
        Client(
            config(provider="bailian", bailian_api_key="fixture"),
            opener=lambda *_a, **_k: io.BytesIO(json.dumps(result).encode()),
        ).transcribe(np.zeros(1600, np.float32))
    assert str(exc.value) == "asr_api_response_invalid"


@pytest.mark.parametrize(
    "url",
    [
        "http://api.example/v1",
        "https://user:secret@api.example",
        "https://api.example?key=secret",
        "https://api.example/#x",
        "https://api.example:0",
        "https://api.example:99999",
        "https://api.example/\n",
    ],
)
def test_endpoints_reject_unsafe_urls(url):
    with pytest.raises(ASRError, match="asr_api_endpoint_invalid"):
        endpoint(url)


@pytest.mark.parametrize(
    "changes,code",
    [
        ({"api_key": ""}, "key_required"),
        ({"api_key": "key\nsecret"}, "key_invalid"),
        ({"model": ""}, "model_required"),
        ({"provider": "missing"}, "provider_unknown"),
        ({"provider": "azure", "azure_api_key": ""}, "key_required"),
        (
            {"provider": "azure", "azure_api_key": "fixture", "azure_region": "bad/region"},
            "region_invalid",
        ),
        (
            {"provider": "azure", "azure_api_key": "fixture", "azure_language": "auto"},
            "language_invalid",
        ),
    ],
)
def test_invalid_configuration_rejected_without_network(changes, code):
    with pytest.raises(ASRError, match="asr_api_" + code):
        Client(config(**changes), opener=lambda *_a, **_k: pytest.fail("unexpected network"))


@pytest.mark.parametrize(
    "payload",
    [
        b"[]",
        b"not json",
        b'{"text":null}',
        b'{"text":false}',
        json.dumps({"text": "x" * 8193}).encode(),
        json.dumps({"text": "x\x00y"}).encode(),
    ],
)
def test_invalid_responses_do_not_escape_as_transcripts(payload):
    client = Client(config(), opener=lambda *_a, **_k: io.BytesIO(payload))
    with pytest.raises(ASRError, match="asr_api_response_invalid"):
        client.transcribe(np.zeros(4000, np.float32))


def test_http_errors_limits_and_timeouts_never_expose_provider_content_or_retry():
    for status in [401, 403, 429, 500]:
        calls = []

        def open(request, *, calls=calls, status=status, **kwargs):
            calls.append(request)
            raise urllib.error.HTTPError(
                "https://private.example/secret",
                status,
                "PRIVATE-RESPONSE",
                {},
                io.BytesIO(b"PRIVATE-KEY-AND-AUDIO"),
            )

        with pytest.raises(ASRError) as exc:
            Client(config(), opener=open).transcribe(np.zeros(4000, np.float32))
        assert str(exc.value) == "asr_api_http_" + str(status) and len(calls) == 1
    with pytest.raises(ASRError, match="asr_api_response_exceeds_limit"):
        Client(
            config(), opener=lambda *_a, **_k: io.BytesIO(b"x" * (MAX_RESPONSE_BYTES + 1))
        ).transcribe(np.zeros(4000, np.float32))

    def timeout(*_a, **_k):
        raise TimeoutError("PRIVATE-URL")

    with pytest.raises(ASRError, match="asr_api_service_unavailable"):
        Client(config(), opener=timeout).transcribe(np.zeros(4000, np.float32))


class Recognizer:
    def __init__(self):
        self.calls = []

    def transcribe(self, samples):
        self.calls.append(samples.copy())
        return "你好 world 日本語", "zh"


def network(**changes):
    recognizer = Recognizer()
    module = NetworkModule(lambda _: recognizer)
    module.prepare(PrepareContext("asr", {"audio": AudioFormat(16000)}, {}), config(**changes))
    module.start(StreamContext())
    return module, recognizer


def send(module, samples, start=0, **kwargs):
    return module.process(
        PortBundle.audio(
            AudioPacket(
                np.ascontiguousarray(samples[:, None]),
                AudioFormat(16000),
                Fraction(start, 16000),
                **kwargs,
            )
        )
    ).ports["text"]


def test_utterance_timestamps_owned_pcm_partial_eos_and_no_silent_uploads():
    module, recognizer = network()
    assert not send(module, np.zeros(16000 * 20, np.float32))
    assert not recognizer.calls and module.frames == 3200
    # Begin a new timeline so the initial pre-roll from the silence test is dropped.
    samples = np.full(4001, 0.1, np.float32)
    assert not send(module, samples, start=400000)
    samples[:] = 1
    out = send(module, np.zeros(8000, np.float32), start=404001)
    segment = TextSegment.from_packet(out[0])
    assert segment.start == 25 and segment.end == Fraction(412001, 16000)
    np.testing.assert_allclose(recognizer.calls[0][:4001], 0.1)
    assert out[0].private and segment.final and segment.language == "zh"
    send(module, np.full(3333, 0.2, np.float32), start=412001)
    tail = module.flush().outputs.ports["text"]
    assert TextSegment.from_packet(tail[0]).end == Fraction(415334, 16000)
    assert not module.flush().outputs.ports and len(recognizer.calls) == 2
    with pytest.raises(ASRError, match="drained"):
        send(module, np.zeros(320, np.float32))


def test_maximum_segment_bound_and_discontinuity_drop_old_voice():
    module, recognizer = network(max_segment_seconds=1)
    out = send(module, np.full(32001, 0.1, np.float32))
    assert len(out) == 2 and [len(x) for x in recognizer.calls] == [16000, 16000]
    send(module, np.full(3333, 0.1, np.float32), start=50000, epoch_id=1)
    send(module, np.full(3333, 0.2, np.float32), start=60000, epoch_id=2)
    tail = module.flush().outputs.ports["text"]
    assert tail[0].epoch_id == 2 and tail[0].observed_start == Fraction(60000, 16000)
    assert len(recognizer.calls[-1]) == 3333 and max(map(len, recognizer.calls)) == 16000
    np.testing.assert_allclose(recognizer.calls[-1], 0.2)


@contextmanager
def server(handler):
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as http:
        thread = Thread(target=http.serve_forever)
        thread.start()
        try:
            yield http
        finally:
            http.shutdown()
            thread.join(2)


def test_redirect_is_not_followed_with_audio_or_credentials():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            calls.append(self.path)
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(307)
            self.send_header("Location", "http://127.0.0.1:1/stolen")
            self.end_headers()

    with server(Handler) as http:
        client = Client(config(base_url=f"http://127.0.0.1:{http.server_port}/v1"))
        with pytest.raises(ASRError, match="asr_api_redirect_rejected"):
            client.transcribe(np.zeros(4000, np.float32))
    assert calls == ["/v1/audio/transcriptions"]


def test_installed_network_worker_has_no_model_or_warmup_and_exits_after_use(tmp_path):
    from obliviousmic.plugin_manager import PluginManager
    from obliviousmic.sdk import ModuleRegistry

    from tools.build_module import build

    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            calls.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"text":"hello","language":"en"}')

    package = build("asr_api", tmp_path / "package", include_models=True, download=False)
    registry = ModuleRegistry()
    manager = PluginManager(tmp_path / "data", registry)
    record = manager.install(package["bundle_path"], trusted=True, official=True)
    assert not record.get("bundled_assets")
    module = registry.get("org.obliviousmic.asr_api").factory()
    with server(Handler) as http:
        try:
            module.prepare(
                PrepareContext("cloud", {"audio": AudioFormat(16000)}, {}),
                config(base_url=f"http://127.0.0.1:{http.server_port}/v1"),
            )
            process = module.worker.worker
            assert "fixture-key" not in " ".join(process.args) and calls == []
            module.start(StreamContext())
            silence = tuple(
                AudioPacket(
                    np.zeros((320, 1), np.float32), AudioFormat(16000), Fraction(i * 320, 16000)
                )
                for i in range(70)
            )
            assert not module.process(PortBundle({"audio": silence})).ports["text"]
            assert calls == []
            send(module, np.full(4000, 0.1, np.float32), start=22400)
            result = module.flush().outputs.ports["text"]
            assert TextSegment.from_packet(result[0]).text == "hello" and len(calls) == 1
        finally:
            module.dispose()
        assert process.poll() is not None


def test_slow_api_keeps_parallel_audio_running_and_buffers_more_than_four_seconds(tmp_path):
    from obliviousmic.__main__ import builtin_registry
    from obliviousmic.graph.dag import DAGCompiler
    from obliviousmic.graph.execution import create_runtime
    from obliviousmic.plugin_manager import PluginManager

    from tools.build_module import build

    entered, release = Event(), Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            entered.set()
            assert release.wait(5)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"text":"hello"}')

    registry = builtin_registry()
    package = build("asr_api", tmp_path / "package", include_models=True, download=False)
    PluginManager(tmp_path / "data", registry).install(
        package["bundle_path"], trusted=True, official=True
    )
    with server(Handler) as http:
        graph = {
            "graph_schema": 1,
            "graph_id": "api",
            "revision": 1,
            "execution": "dag",
            "nodes": [
                {"id": "mic", "type": "core.source"},
                {"id": "speaker", "type": "core.sink"},
                {
                    "id": "asr",
                    "type": "org.obliviousmic.asr_api",
                    "settings": config(
                        base_url=f"http://127.0.0.1:{http.server_port}/v1", max_segment_seconds=1
                    ),
                },
                {"id": "captions", "type": "core.text_sink"},
            ],
            "edges": [
                {"from": a, "to": b}
                for a, b in [
                    ("mic.audio", "speaker.audio"),
                    ("mic.audio", "asr.audio"),
                    ("asr.text", "captions.text"),
                ]
            ],
        }
        runtime = create_runtime(
            DAGCompiler(registry).compile(graph, AudioFormat(16000)), mode="realtime"
        )
        try:
            runtime.prepare()
            runtime.start()
            assert not entered.is_set() and runtime.limit >= 576000
            for start in range(0, 16000, 320):
                runtime.process_endpoints(
                    {
                        "mic": PortBundle.audio(
                            AudioPacket(
                                np.full((320, 1), 0.1, np.float32),
                                AudioFormat(16000),
                                Fraction(start, 16000),
                            )
                        )
                    }
                )
            deadline = monotonic() + 3
            while not entered.is_set():
                assert monotonic() < deadline
                runtime.process_endpoints({})
                sleep(0.001)
            for start in range(16000, 160000, 320):
                out = runtime.process_endpoints(
                    {
                        "mic": PortBundle.audio(
                            AudioPacket(
                                np.full((320, 1), 0.1, np.float32),
                                AudioFormat(16000),
                                Fraction(start, 16000),
                            )
                        )
                    }
                )
                np.testing.assert_allclose(out["speaker"].ports["audio"][0].samples, 0.1)
            assert sum(p.frames for b in runtime.pending["asr"] for p in b.ports["audio"]) > 65536
            release.set()
            out = runtime.flush_endpoints()
            assert len(out["captions"].ports["text"]) == 10
        finally:
            release.set()
            runtime.close()
