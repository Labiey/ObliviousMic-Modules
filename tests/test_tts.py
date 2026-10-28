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
    value["qwen_api_key"] = "fixture-key"
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


@pytest.mark.parametrize(
    "rate,pitch,expected",
    [(0.5, 0.5, "-50%"), (1.15, 1.2, "+20%"), (2, 1.5, "+50%"), (1, 1.55, "+55%"), (1, 2, "+100%")],
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
        ("azure_pitch", 2.01),
        ("azure_pitch", float("nan")),
        ("azure_rate", True),
    ],
)
def test_azure_rejects_invalid_prosody_before_network(field, value):
    with pytest.raises(TTSError, match="tts_azure_prosody_invalid"):
        Client(config(provider="azure", azure_api_key="fixture", **{field: value}))


def test_pitch_table_can_save_155_and_two_without_changing_defaults(tmp_path):
    from obliviousmic.presets import PresetStore
    from obliviousmic.sdk import Manifest, ModuleRegistration, ModuleRegistry

    registry = ModuleRegistry()
    registry.register(ModuleRegistration(Manifest(Module.MANIFEST), Module.SCHEMA, Module))
    store = PresetStore(tmp_path, registry)
    for pitch in (1.55, 2.0):
        saved = store.create(
            "org.obliviousmic.tts", str(pitch), {"provider": "azure", "azure_pitch": pitch}
        )
        assert (
            store.get("org.obliviousmic.tts", saved["preset_id"])["settings"]["azure_pitch"]
            == pitch
        )
    assert config()["azure_pitch"] == 1.0


@pytest.mark.parametrize(
    "change,code",
    [
        ({"provider": "qwen", "qwen_api_key": ""}, "key_required"),
        ({"provider": "azure"}, "key_required"),
    ],
)
def test_required_parameters_are_checked(change, code):
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


@pytest.mark.parametrize("provider", ["qwen", "azure"])
def test_cloud_prepare_never_sends_a_billable_warmup_request(provider):
    def forbidden(*args, **kwargs):
        raise AssertionError("network access during cloud prepare")

    module = NetworkModule(lambda config: Client(config, forbidden))
    module.prepare(
        PrepareContext("tts", {}, {"audio": AudioFormat(24000)}),
        config(provider=provider, qwen_api_key="fixture", azure_api_key="fixture"),
    )
    module.dispose()


def test_cloud_only_schema_and_permissions():
    assert Module.SCHEMA["properties"]["provider"]["enum"] == ["qwen", "azure"]
    assert Module.MANIFEST["permissions"] == []
    assert not set(Module.SCHEMA["properties"]) & {
        "profile_id",
        "local_endpoint",
        "local_install_directory",
        "auto_start_local",
        "reference_path",
        "reference_prompt",
        "reference_language",
    }
    with pytest.raises(TTSError, match="provider_unknown"):
        Client(config(provider="removed-provider"))
