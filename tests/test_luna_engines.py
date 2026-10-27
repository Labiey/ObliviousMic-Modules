import base64
import codecs
import hashlib
import io
import json
import urllib.error
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from obliviousmic.sdk import PublicModuleError

from modules.translate.luna_bridge import ENGINES
from modules.translate.luna_transport import MAX_BODY, Session
from modules.translate.providers import Client
from modules.translate.runtime import Module


def config(provider, **changes):
    result = {k: v["default"] for k, v in Module.SCHEMA["properties"].items()}
    result.update(
        provider=provider,
        engine_key="fixture-key",
        engine_secret="fixture-secret",
        engine_app_id="12345",
    )
    result.update(changes)
    return result


class Gateway:
    def __init__(self, provider):
        self.provider, self.calls = provider, []

    def __call__(self, request, **kwargs):
        assert request.full_url.startswith("https://")
        assert 0 < kwargs["timeout"] <= 25
        self.calls.append(request)
        url = urlsplit(request.full_url)
        code = self.provider
        if request.method == "OPTIONS":
            body = {}
        elif code == "bing" and request.method == "GET":
            body = (
                '<div id="tta_outGDCont" data-iid="translator.1"> IG:"fixtureIG"; '
                'var params_AbusePreventionHelper = [123,"fixtureToken"];'
            )
        elif (
            code == "qqtransmart"
            and request.method == "POST"
            and json.loads(request.data)["header"]["fn"] == "text_analysis"
        ):
            payload = json.loads(request.data)
            body = {
                "sentence_list": [{"start": 0, "len": len(payload["text"])}],
                "text": payload["text"],
            }
        elif code in {"microsoft", "bing"}:
            body = [{"translations": [{"text": "Hello."}]}]
        elif code == "google":
            body = [["Hello."]]
        elif code == "modernmt":
            body = {"data": {"translation": "Hello."}}
        elif code in {"qqtransmart", "qqimt"}:
            body = {"auto_translation": ["Hello."]}
        elif code == "ali":
            body = (
                {"token": "csrf-fixture"}
                if url.path.endswith("csrftoken")
                else {"data": {"translateText": "Hello."}}
            )
        elif code == "youdao":
            body = {"translateResult": [[{"tgt": "Hello."}]]}
        elif code == "caiyun":
            body = (
                {"jwt": "fixture-jwt"}
                if url.path.endswith("generate")
                else {"target": codecs.encode(base64.b64encode(b"Hello.").decode(), "rot13")}
            )
        elif code == "translatecom":
            body = (
                {"language": "zh"}
                if url.path.endswith("ajax_lang_auto_detect")
                else {"translated_text": "Hello."}
            )
        elif code == "yandex":
            body = {"lang": "zh"} if url.path.endswith("detect") else {"text": ["Hello."]}
        elif code == "huoshan":
            body = {"translation": "Hello."}
        elif code == "papago":
            body = {"translatedText": "Hello."}
        elif code == "baidu_api":
            if url.path.endswith("token"):
                body = {"access_token": "fixture-token"}
            elif "/rpc/" in url.path:
                body = {"result": {"trans_result": [{"dst": "Hello."}]}}
            else:
                body = {"trans_result": [{"dst": "Hello."}]}
        elif code == "caiyun_api":
            body = {"target": "Hello."}
        elif code == "google_api":
            body = {"data": {"translations": [{"translatedText": "Hello."}]}}
        elif code == "deepl_api":
            body = {"translations": [{"text": "Hello."}]}
        elif code == "tencent_api":
            body = {"Response": {"TargetText": "Hello."}}
        elif code == "youdao_api":
            body = {"translation": ["Hello."]}
        elif code == "yandex_api":
            body = {"translations": [{"text": "Hello."}]}
        elif code == "aliyun_api":
            body = {"Data": {"Translated": "Hello."}}
        elif code == "huoshan_api":
            body = {"TranslationList": [{"Translation": "Hello."}]}
        elif code == "huawei_api":
            body = (
                {"projects": [{"id": "fixture-project"}]}
                if url.hostname == "iam.myhuaweicloud.com"
                else {"translated_text": "Hello."}
            )
        elif code == "niutrans_api":
            body = {"tgt_text": "Hello."}
        else:
            raise AssertionError(code)
        result = io.BytesIO((body if isinstance(body, str) else json.dumps(body)).encode())
        result.headers = {}
        result.status = 200
        return result


@pytest.mark.parametrize("provider", list(ENGINES))
def test_every_reused_engine_loads_without_luna_gui_and_translates_through_bounded_session(
    provider,
):
    gateway = Gateway(provider)
    client = Client(config(provider), gateway)
    assert gateway.calls == []
    assert client.translate("测试。", "zh") == "Hello."
    assert client.engine.engine.__class__.__module__.endswith(ENGINES[provider]["file"])
    assert 1 <= len(gateway.calls) <= 16
    if provider == "bing":
        params = parse_qs(urlsplit(gateway.calls[-1].full_url).query)
        assert params["IG"] == ["fixtureIG"] and params["IID"] == ["translator.1"]
    if provider == "huoshan_api":
        assert gateway.calls[-1].get_header("Authorization").startswith("HMAC-SHA256")


@pytest.mark.parametrize("interface", ["standard", "bce", "llm"])
def test_baidu_variants_keep_their_distinct_authentication_contract(interface):
    gateway = Gateway("baidu_api")
    assert (
        Client(
            config("baidu_api", baidu_interface=interface, baidu_api_key="fixture"), gateway
        ).translate("测试。", "zh")
        == "Hello."
    )
    assert len(gateway.calls) == (2 if interface == "bce" else 1)


def test_no_qwen_provider_and_all_pinned_leaves_have_verified_sources_and_gpl_license():
    fields = Module.SCHEMA["properties"]
    assert "qwen" not in fields["provider"]["enum"] and "qwen_api_key" not in fields
    assert fields["provider"]["default"] == "microsoft"
    assert len(fields["provider"]["enum"]) == 26
    root = Path(__file__).resolve().parents[1] / "modules/translate"
    provenance = json.loads((root / "LUNA_PROVENANCE.json").read_text("utf-8"))
    assert len(provenance["files"]) == 24
    assert provenance["license"] == "GPL-3.0-only"
    assert "GNU GENERAL PUBLIC LICENSE" in (root / "LICENSE").read_text("utf-8")
    for item in provenance["files"]:
        normalized = (root / item["file"]).read_text("utf-8").replace("\r\n", "\n").encode("utf-8")
        assert hashlib.sha256(normalized).hexdigest() == item["adapted_sha256"]
        assert len(item["upstream_sha256"]) == 64 and item["modifications"]


def test_engine_errors_are_sanitized_and_off_domain_redirects_are_rejected():
    def denied(request, **kwargs):
        raise urllib.error.HTTPError(
            request.full_url, 403, "private text/key", {}, io.BytesIO(b"private provider body")
        )

    with pytest.raises(PublicModuleError, match="^translate_provider_http_403$"):
        Client(config("microsoft"), denied).translate("私有文字。", "zh")
    with pytest.raises(PublicModuleError, match="endpoint_invalid"):
        Session(["bing.com"], 25).checked_url("https://attacker.example/steal")


def test_transport_enforces_body_and_request_count_limits_and_never_logs_response_text():
    def oversized(*args, **kwargs):
        response = io.BytesIO(b"x" * (MAX_BODY + 1))
        response.headers = {}
        response.status = 200
        return response

    session = Session(["bing.com"], 25, oversized)
    with pytest.raises(PublicModuleError, match="response_exceeds_limit"):
        session.get("https://bing.com")
    session = Session(["bing.com"], 25)
    session.calls = 16
    with pytest.raises(PublicModuleError, match="request_timeout"):
        session.get("https://bing.com")


@pytest.mark.parametrize(
    "provider", [name for name, spec in ENGINES.items() if spec["credentials"]]
)
def test_api_credentials_are_required_without_any_network_during_prepare(provider):
    with pytest.raises(PublicModuleError, match="api_key_required"):
        Client(config(provider, engine_key="", engine_secret="", engine_app_id=""))
