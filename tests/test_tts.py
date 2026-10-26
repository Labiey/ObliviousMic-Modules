import io
import json
import urllib.error
import xml.etree.ElementTree as ET
from fractions import Fraction

import numpy as np
import pytest
import soundfile as sf
from obliviousmic.sdk import AudioFormat, PortBundle, PrepareContext, StreamContext, TextSegment

from modules.tts.network import Module as NetworkModule
from modules.tts.providers import Client, NoRedirects, TTSError
from modules.tts.runtime import Module


def config(**change):
    value = {k: p["default"] for k, p in Module.SCHEMA["properties"].items()}
    value.update(change)
    return value


def wav(rate=24000, frames=2401, value=0.2):
    output = io.BytesIO()
    sf.write(output, np.full((frames, 1), value, np.float32), rate, format="WAV", subtype="FLOAT")
    return output.getvalue()


def test_qwen_contract_uses_separate_authenticated_request_and_uncredentialed_https_audio():
    calls = []

    def open(request, **kwargs):
        calls.append(request)
        if len(calls) == 1:
            return io.BytesIO(
                json.dumps(
                    {
                        "output": {
                            "audio": {
                                "url": "http://result.oss-cn-beijing.aliyuncs.com/audio.wav?Signature=fixture"
                            }
                        }
                    }
                ).encode()
            )
        return io.BytesIO(wav())

    result = Client(config(provider="qwen", qwen_api_key="fixture-key"), open).synthesize(
        "你好", "zh"
    )
    assert result.shape == (2401, 1)
    payload = json.loads(calls[0].data)
    assert payload["input"]["language_type"] == "Chinese" and payload["input"]["voice"] == "Cherry"
    assert calls[0].get_header("Authorization") == "Bearer fixture-key"
    assert calls[1].full_url.startswith("https://") and calls[1].get_header("Authorization") is None


def test_azure_contract_escapes_ssml_and_selects_language_voice():
    calls = []

    def open(request, **kwargs):
        calls.append(request)
        return io.BytesIO(wav(48000, 4802))

    audio = Client(config(provider="azure", azure_api_key="fixture-key"), open).synthesize(
        "Hello <world> & test", "en"
    )
    root = ET.fromstring(calls[0].data)
    voice = root.find("{http://www.w3.org/2001/10/synthesis}voice")
    assert voice.text == "Hello <world> & test" and voice.attrib["name"] == "en-US-JennyNeural"
    assert calls[0].get_header("Ocp-apim-subscription-key") == "fixture-key"
    assert len(audio) == 2401


def test_gpt_sovits_contract_contains_explicit_local_reference_and_decodes_audio():
    calls = []

    def open(request, **kwargs):
        calls.append(request)
        return io.BytesIO(wav(32000, 3201))

    audio = Client(
        config(reference_path="synthetic-reference.wav", reference_prompt="test"), open
    ).synthesize("こんにちは", "ja")
    body = json.loads(calls[0].data)
    assert body["text_lang"] == "ja" and body["ref_audio_path"] == "synthetic-reference.wav"
    assert body["streaming_mode"] is False and body["media_type"] == "wav"
    assert abs(len(audio) - 3201 * 24000 / 32000) <= 1


@pytest.mark.parametrize(
    "rate,pitch,expected", [(0.5, 0.5, "-50%"), (1.15, 1.2, "+20%"), (2, 1.5, "+50%")]
)
def test_azure_prosody_preserves_literal_text_and_maps_multipliers(rate, pitch, expected):
    calls = []

    def open(request, **kwargs):
        calls.append(request)
        return io.BytesIO(wav())

    Client(
        config(provider="azure", azure_api_key="fixture", azure_rate=rate, azure_pitch=pitch), open
    ).synthesize("Hello <world> & test", "en")
    root = ET.fromstring(calls[0].data)
    prosody = root.find(
        "{http://www.w3.org/2001/10/synthesis}voice/{http://www.w3.org/2001/10/synthesis}prosody"
    )
    assert prosody.attrib == {"rate": format(rate, ".6g"), "pitch": expected}
    assert prosody.text == "Hello <world> & test"
    assert len(calls) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("azure_rate", 0.49),
        ("azure_rate", 2.01),
        ("azure_pitch", 1.55),
        ("azure_pitch", float("nan")),
        ("azure_rate", True),
    ],
)
def test_azure_rejects_invalid_prosody_before_network(field, value):
    with pytest.raises(TTSError, match="tts_azure_prosody_invalid"):
        Client(config(provider="azure", azure_api_key="fixture", **{field: value}))


@pytest.mark.parametrize(
    "change,code",
    [
        ({"provider": "qwen"}, "key_required"),
        ({"provider": "azure"}, "key_required"),
        ({"local_endpoint": "http://attacker.invalid:9880"}, "loopback"),
        ({"local_endpoint": "http://127.0.0.1:99999"}, "loopback"),
    ],
)
def test_required_parameters_and_local_endpoint_are_checked(change, code):
    with pytest.raises(TTSError, match=code):
        Client(config(**change))


def test_credential_requests_do_not_redirect_or_retry_and_errors_do_not_echo_body():
    with pytest.raises(urllib.error.HTTPError):
        NoRedirects().redirect_request(
            urllib.request.Request("https://example.com"),
            None,
            302,
            "moved",
            {},
            "https://attacker.invalid",
        )
    calls = []

    def open(request, **kwargs):
        calls.append(True)
        raise urllib.error.HTTPError(
            request.full_url, 401, "fixture-secret-prompt", {}, io.BytesIO(b"fixture-key")
        )

    with pytest.raises(TTSError) as error:
        Client(config(provider="azure", azure_api_key="fixture-key"), open).synthesize(
            "private text"
        )
    assert str(error.value) == "tts_provider_http_401" and calls == [True]


@pytest.mark.parametrize(
    "body",
    [b"invalid audio", wav(frames=0), wav(value=float("nan"))],
    ids=["malformed", "empty", "nonfinite"],
)
def test_bad_provider_audio_is_rejected(body):
    client = Client(config(), lambda *a, **k: io.BytesIO(body))
    with pytest.raises(TTSError, match="audio_response_invalid"):
        client.synthesize("test")


def test_generated_audio_clock_is_independent_of_source_text_and_final_ids_are_spoken_once():
    class Fake:
        def __init__(self, config):
            self.calls = []

        def warmup(self):
            pass

        def synthesize(self, text, language):
            self.calls.append(text)
            return np.full((2401, 1), 0.1, np.float32)

    module = NetworkModule(Fake)
    module.prepare(PrepareContext("tts", {}, {"audio": AudioFormat(24000)}), config())
    module.start(StreamContext())
    first = TextSegment("one", "first", Fraction(10), Fraction(12), "en", final=False)
    assert not module.process(PortBundle({"text": (first.to_packet("input"),)})).ports["audio"]
    first = TextSegment("one", "first", Fraction(10), Fraction(12), "en")
    second = TextSegment("two", "second", Fraction(20), Fraction(30), "en")
    outputs = []
    for segment in (first, first, second):
        outputs.extend(
            module.process(PortBundle({"text": (segment.to_packet("input"),)})).ports["audio"]
        )
    assert module.client.calls == ["first", "second"]
    assert sum(p.frames for p in outputs) == 4802 and outputs[0].media_start == 0
    assert outputs[-1].media_end == Fraction(4802, 24000)
    assert all("generated" in p.flags for p in outputs)
    module.reset("epoch")
    module.start(StreamContext("main", 1))
    result = module.process(
        PortBundle({"text": (first.to_packet("input", StreamContext("main", 1)),)})
    )
    assert result.ports["audio"][0].media_start == 0 and result.ports["audio"][0].epoch_id == 1
    module.dispose()


def test_local_warmup_has_separate_deadline_and_restores_steady_state_settings():
    calls = []

    def open(request, **kwargs):
        calls.append((json.loads(request.data), kwargs["timeout"]))
        return io.BytesIO(wav())

    client = Client(config(), open)
    original = client.config
    client.warmup()
    assert len(calls) == 1 and calls[0][0]["text"] == "你好。"
    assert 89 < calls[0][1] <= 90
    assert client.config is original and client.config["request_timeout_seconds"] == 25
    client.synthesize("正常请求")
    assert calls[1][1] <= 25


@pytest.mark.parametrize("provider", ["qwen", "azure"])
def test_cloud_prepare_never_sends_a_billable_warmup_request(provider):
    def forbidden(*args, **kwargs):
        raise AssertionError("network access during cloud prepare")

    Client(
        config(provider=provider, qwen_api_key="fixture", azure_api_key="fixture"), forbidden
    ).warmup()


def test_failed_local_warmup_restores_configuration():
    def failed(*args, **kwargs):
        raise TimeoutError()

    client = Client(config(), failed)
    original = client.config
    with pytest.raises(TTSError, match="unavailable_or_timeout"):
        client.warmup()
    assert client.config is original


def test_loopback_model_requests_and_discovery_do_not_use_environment_proxy(monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    from modules.tts.local_service import probe

    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = json.dumps(
                {"paths": {p: {} for p in ["/tts", "/set_gpt_weights", "/set_sovits_weights"]}}
            ).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(wav())

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = Thread(target=server.serve_forever)
        worker.start()
        try:
            monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
            monkeypatch.setenv("no_proxy", "")
            endpoint = f"http://127.0.0.1:{server.server_port}"
            assert probe(endpoint)
            result = Client(config(local_endpoint=endpoint)).synthesize("private local text")
            assert len(result) == 2401 and len(received) == 1
        finally:
            server.shutdown()
            worker.join(2)
        assert not worker.is_alive()
