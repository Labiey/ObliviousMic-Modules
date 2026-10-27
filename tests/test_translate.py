import io
import json
import urllib.error
import urllib.request
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from obliviousmic.sdk import PortBundle, PrepareContext, StreamContext, TextSegment

from modules.translate.network import Module as NetworkModule
from modules.translate.providers import Client, NoRedirects, TranslationError, endpoint
from modules.translate.runtime import Module


def config(**changes):
    result = {key: p["default"] for key, p in Module.SCHEMA["properties"].items()}
    result.update(qwen_api_key="synthetic-key")
    result.update(changes)
    return result


def response(text="Hello.", reason="stop"):
    return io.BytesIO(
        json.dumps({"choices": [{"message": {"content": text}, "finish_reason": reason}]}).encode()
    )


@pytest.mark.parametrize(
    "region,host",
    [("beijing", "dashscope.aliyuncs.com"), ("singapore", "dashscope-intl.aliyuncs.com")],
)
def test_qwen_protocol_separates_instructions_and_private_text_and_preparation_is_free(
    region, host
):
    calls = []

    def open(request, **kwargs):
        calls.append(request)
        assert kwargs["timeout"] == 25
        return response()

    client = Client(config(qwen_region=region), open)
    assert not calls
    assert client.translate("你好。", "zh") == "Hello."
    assert calls[0].full_url == f"https://{host}/compatible-mode/v1/chat/completions"
    assert calls[0].get_header("Authorization") == "Bearer synthetic-key"
    body = json.loads(calls[0].data)
    assert body["messages"][1] == {"role": "user", "content": "你好。"}
    assert "English" in body["messages"][0]["content"]
    assert not body["stream"] and body["temperature"] == 0


def test_azure_contract_uses_translator_not_speech_and_omits_auto_from_and_global_region():
    calls = []

    def open(request, **kwargs):
        calls.append(request)
        return io.BytesIO(b'[{"translations":[{"text":"Hello."}]}]')

    client = Client(config(provider="azure", azure_api_key="translator-fixture"), open)
    assert client.translate("你好。") == "Hello."
    assert (
        calls[0].full_url
        == "https://api.cognitive.microsofttranslator.com/translate?api-version=3.0&to=en"
    )
    assert calls[0].get_header("Ocp-apim-subscription-key") == "translator-fixture"
    assert calls[0].get_header("Ocp-apim-subscription-region") is None
    assert json.loads(calls[0].data) == [{"Text": "你好。"}]
    regional = Client(
        config(
            provider="azure", azure_api_key="fixture", azure_region="eastasia", target_language="zh"
        ),
        open,
    )
    regional.translate("Hello.", "en")
    assert "to=zh-Hans&from=en" in calls[-1].full_url
    assert calls[-1].get_header("Ocp-apim-subscription-region") == "eastasia"


@pytest.mark.parametrize(
    "url",
    [
        "http://remote.example/v1",
        "https://user:pass@example.com",
        "https://example.com/?key=private",
        "https://example.com/#key",
        "https://example.com:99999",
        "file:///tmp/test",
    ],
)
def test_endpoints_reject_insecure_cloud_urls_and_embedded_credentials(url):
    with pytest.raises(TranslationError, match="endpoint_invalid"):
        endpoint(url)


def test_local_key_optional_cloud_key_required_and_same_language_bypasses_network():
    Client(config(provider="openai_compatible", api_key=""))
    with pytest.raises(TranslationError, match="api_key_required"):
        Client(config(provider="openai_compatible", base_url="https://example.com/v1", api_key=""))
    client = Client(config(), lambda *_a, **_k: pytest.fail("unexpected network request"))
    assert client.translate("Already English.", "en-US") == "Already English."


@pytest.mark.parametrize("status", [401, 403, 429, 503])
def test_public_errors_never_contain_provider_body_text_or_keys(status):
    def open(request, **kwargs):
        raise urllib.error.HTTPError(
            request.full_url, status, "secret provider body", {}, io.BytesIO(b"private text/key")
        )

    with pytest.raises(TranslationError, match=f"^translate_provider_http_{status}$"):
        Client(config(), open).translate("私有文字", "zh")
    with pytest.raises(urllib.error.HTTPError, match="redirect rejected"):
        request = urllib.request.Request("https://original.example")
        NoRedirects().redirect_request(request, None, 302, "private", {}, "https://other.example")


@pytest.mark.parametrize(
    "body", [b"not json", b"{}", b'{"choices":[]}', b'{"choices":[{"message":{"content":null}}]}']
)
def test_malformed_empty_or_nontext_responses_fail_closed(body):
    with pytest.raises(TranslationError, match="translate_response_invalid"):
        Client(config(), lambda *_a, **_k: io.BytesIO(body)).translate("测试", "zh")


def test_limits_and_truncation_are_explicit_and_requests_never_retry():
    with pytest.raises(TranslationError, match="response_incomplete"):
        Client(config(), lambda *_a, **_k: response("partial", "length")).translate("你好", "zh")
    with pytest.raises(TranslationError, match="response_exceeds_limit"):
        Client(config(), lambda *_a, **_k: io.BytesIO(b"x" * (256 * 1024 + 1))).translate(
            "你好", "zh"
        )
    calls = []

    def timeout(*_a, **_k):
        calls.append(1)
        raise TimeoutError("private")

    with pytest.raises(TranslationError, match="service_unavailable"):
        Client(config(), timeout).translate("你好", "zh")
    assert len(calls) == 1
    with pytest.raises(TranslationError, match="text_limit"):
        Client(config(maximum_text_characters=2), timeout).translate("太长的文字", "zh")
    assert len(calls) == 1


def test_revisions_timestamps_language_privacy_and_epoch_reset():
    calls = []

    class FakeClient:
        def __init__(self, config):
            pass

        def translate(self, text, language):
            calls.append(text)
            return "translated " + text

    module = NetworkModule(FakeClient)
    module.prepare(PrepareContext("translator", {}, {}, "realtime"), config(final_only=False))
    module.start(StreamContext("speech", 2))

    def send(revision, final, epoch=2, text="test"):
        packet = TextSegment(
            "one", text, Fraction(5), Fraction(7), "ja", final, revision
        ).to_packet("asr", StreamContext("speech", epoch))
        return module.process(PortBundle({"text": (packet,)})).ports["text"]

    assert send(0, False)
    assert not send(0, False)
    assert send(0, True)  # Finalization may share a revision with the last partial.
    assert not send(1, False)  # A finalized segment cannot regress to a partial.
    assert not send(0, True)
    packet = send(2, True)[0]
    translated = TextSegment.from_packet(packet)
    assert translated.start == 5 and translated.end == 7 and translated.language == "en"
    assert translated.segment_id == "one" and translated.revision == 2 and translated.final
    assert packet.private and packet.source_instance == "translator" and packet.epoch_id == 2
    assert send(0, True, 3)
    module.reset("new stream")
    assert send(0, True)
    module.prepare(PrepareContext("translator", {}, {}, "realtime"), config())
    assert not send(0, False)
    assert send(0, True)
    for index in range(270):
        send(0, True, epoch=index + 10)
    assert len(module.seen) == 256


def test_real_ipc_worker_and_loopback_transport_ignore_proxy_and_clean_up(monkeypatch):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            assert self.path == "/v1/chat/completions"
            assert self.headers.get("Authorization") is None
            self.send_response(200)
            self.end_headers()
            self.wfile.write(response("こんにちは。").getvalue())

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        module = Module()
        try:
            monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
            monkeypatch.setenv("no_proxy", "")
            module.prepare(
                PrepareContext("tr", {}, {}, "realtime"),
                config(
                    provider="openai_compatible",
                    base_url=f"http://127.0.0.1:{server.server_port}/v1",
                    model="local-fixture",
                    target_language="ja",
                ),
            )
            assert not received
            process = module.worker.worker
            module.start(StreamContext("typed", 3))
            packet = TextSegment("one", "Hello", Fraction(0), Fraction(1), "en").to_packet(
                "keyboard", StreamContext("typed", 3)
            )
            result = module.process(PortBundle({"text": (packet,)}))
            assert TextSegment.from_packet(result.ports["text"][0]).text == "こんにちは。"
            assert len(received) == 1
            module.stop()
        finally:
            module.dispose()
            server.shutdown()
            thread.join(2)
        assert process.poll() is not None


def test_installed_translation_graph_keeps_audio_route_running_while_network_is_pending(tmp_path):
    from threading import Event
    from time import monotonic, sleep

    import numpy as np
    from obliviousmic.graph.dag import DAGCompiler
    from obliviousmic.graph.dag_runtime import DAGRuntime
    from obliviousmic.graph.primitives import register_primitives
    from obliviousmic.plugin_manager import PluginManager
    from obliviousmic.sdk import AudioFormat, AudioPacket, ModuleRegistry

    from tools.build_module import build

    entered, release = Event(), Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            entered.set()
            assert release.wait(3)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(response("Hello.").getvalue())

    registry = ModuleRegistry()
    register_primitives(registry)
    package = build("translate", tmp_path / "package", include_models=True, download=False)
    PluginManager(tmp_path / "data", registry).install(
        package["bundle_path"], trusted=True, official=True
    )
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            document = {
                "graph_schema": 1,
                "graph_id": "translation-test",
                "revision": 1,
                "execution": "dag",
                "nodes": [
                    {"id": "keyboard", "type": "core.text_source"},
                    {
                        "id": "translator",
                        "type": "org.obliviousmic.translate",
                        "settings": config(
                            provider="openai_compatible",
                            base_url=f"http://127.0.0.1:{server.server_port}/v1",
                        ),
                    },
                    {"id": "captions", "type": "core.text_sink"},
                    {"id": "mic", "type": "core.source"},
                    {"id": "speaker", "type": "core.sink"},
                ],
                "edges": [
                    {"from": a, "to": b}
                    for a, b in [
                        ("keyboard.text", "translator.text"),
                        ("translator.text", "captions.text"),
                        ("mic.audio", "speaker.audio"),
                    ]
                ],
            }
            fmt = AudioFormat(16000)
            with DAGRuntime(
                DAGCompiler(registry).compile(document, fmt), mode="realtime"
            ) as runtime:
                assert not entered.is_set()
                text = TextSegment(
                    "from-asr-or-keyboard", "你好。", Fraction(1), Fraction(2), "zh"
                ).to_packet("keyboard")
                assert not runtime.process_endpoints({"keyboard": PortBundle({"text": (text,)})})
                assert entered.wait(2)
                pcm = AudioPacket(np.full((160, 1), 0.25, np.float32), fmt, Fraction(0))
                audio = runtime.process_endpoints({"mic": PortBundle.audio(pcm)})
                np.testing.assert_array_equal(
                    audio["speaker"].ports["audio"][0].samples, pcm.samples
                )
                release.set()
                outputs = {}
                deadline = monotonic() + 3
                while "captions" not in outputs:
                    assert monotonic() < deadline
                    outputs.update(runtime.process_endpoints({}))
                    sleep(0.005)
                result = TextSegment.from_packet(outputs["captions"].ports["text"][0])
                assert (
                    result.text == "Hello."
                    and result.start == 1
                    and result.end == 2
                    and result.language == "en"
                )
        finally:
            release.set()
            server.shutdown()
            thread.join(2)
