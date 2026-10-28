# SPDX-License-Identifier: Apache-2.0
"""Short audio transcription protocols; bounded requests, no retries or redirects."""

import io
import json
import re
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
import wave

import numpy as np
from obliviousmic.sdk import PublicModuleError

RATE = 16000
MAX_RESPONSE_BYTES = 256 * 1024
MAX_TEXT = 8192
LANGUAGES = {"auto", "zh", "en", "ja"}
AZURE_LANGUAGES = {"zh-CN": "zh", "en-US": "en", "ja-JP": "ja"}


class ASRError(PublicModuleError):
    pass


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        fp.close()
        raise ASRError("asr_api_redirect_rejected")


def endpoint(value):
    try:
        parsed = urllib.parse.urlsplit(value)
        local = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        if (
            not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.scheme not in ({"http", "https"} if local else {"https"})
            or (parsed.port is not None and not 1 <= parsed.port <= 65535)
            or any(ord(c) <= 32 or ord(c) > 126 for c in value)
        ):
            raise ValueError()
    except ValueError:
        raise ASRError("asr_api_endpoint_invalid") from None
    return value.rstrip("/"), local


def validate_config(config):
    if config["provider"] not in {"openai_compatible", "azure"}:
        raise ASRError("asr_api_provider_unknown")
    if config["provider"] == "openai_compatible":
        _, local = endpoint(config["base_url"])
        if not local and not config["api_key"].strip():
            raise ASRError("asr_api_key_required")
        if not config["model"].strip():
            raise ASRError("asr_api_model_required")
        if config["language"] not in LANGUAGES:
            raise ASRError("asr_api_language_invalid")
        keys = ["api_key"]
    else:
        if not config["azure_api_key"].strip():
            raise ASRError("asr_api_key_required")
        if not re.fullmatch(r"[a-z][a-z0-9-]{1,40}", config["azure_region"]):
            raise ASRError("asr_api_region_invalid")
        if config["azure_language"] not in AZURE_LANGUAGES:
            raise ASRError("asr_api_language_invalid")
        if config["azure_endpoint"]:
            endpoint(config["azure_endpoint"])
        keys = ["azure_api_key"]
    if any(any(ord(c) < 33 or ord(c) > 126 for c in config[key]) for key in keys):
        raise ASRError("asr_api_key_invalid")


def wav_bytes(samples):
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 1 or not 0 < len(x) <= 12 * RATE or not np.isfinite(x).all():
        raise ASRError("asr_api_audio_invalid")
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    result = io.BytesIO()
    with wave.open(result, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(pcm)
    return result.getvalue()


def multipart(audio, fields):
    boundary = "obmic-asr-" + secrets.token_hex(16)
    chunks = []
    for name, value in fields.items():
        chunks.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
            ).encode()
        )
    chunks.extend(
        [
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                'filename="audio.wav"\r\nContent-Type: audio/wav\r\n\r\n'
            ).encode(),
            audio,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), "multipart/form-data; boundary=" + boundary


def checked_text(value):
    if not isinstance(value, str) or len(value) > MAX_TEXT or "\x00" in value:
        raise ASRError("asr_api_response_invalid")
    return value.strip()


class Client:
    def __init__(self, config, opener=None):
        validate_config(config)
        self.config = dict(config)
        url = (
            config["base_url"]
            if config["provider"] == "openai_compatible"
            else (
                config["azure_endpoint"]
                or "https://" + config["azure_region"] + ".stt.speech.microsoft.com"
            )
        )
        _, local = endpoint(url)
        handlers = [NoRedirects()]
        if local:
            handlers.append(urllib.request.ProxyHandler({}))
        self.open = opener or urllib.request.build_opener(*handlers).open

    def _request(self, url, body, headers):
        deadline = time.monotonic() + self.config["request_timeout_seconds"]
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Accept": "application/json", "User-Agent": "ObliviousMic-ASR/1", **headers},
        )
        try:
            with self.open(request, timeout=self.config["request_timeout_seconds"]) as response:
                chunks, total = [], 0
                while True:
                    if time.monotonic() >= deadline:
                        raise ASRError("asr_api_request_timeout")
                    data = getattr(response, "read1", response.read)(
                        min(16384, MAX_RESPONSE_BYTES + 1 - total)
                    )
                    if not data:
                        break
                    total += len(data)
                    if total > MAX_RESPONSE_BYTES:
                        raise ASRError("asr_api_response_exceeds_limit")
                    chunks.append(data)
                result = json.loads(b"".join(chunks))
                if not isinstance(result, dict):
                    raise ValueError()
                return result
        except urllib.error.HTTPError as exc:
            code = exc.code
            exc.close()
            raise ASRError("asr_api_http_" + str(code)) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise ASRError("asr_api_service_unavailable") from None
        except (ValueError, UnicodeError) as exc:
            if isinstance(exc, ASRError):
                raise
            raise ASRError("asr_api_response_invalid") from None

    def transcribe(self, samples):
        config = self.config
        audio = wav_bytes(samples)
        if config["provider"] == "openai_compatible":
            url, _ = endpoint(config["base_url"])
            if not url.endswith("/audio/transcriptions"):
                url += "/audio/transcriptions"
            fields = {"model": config["model"], "response_format": "json"}
            if config["language"] != "auto":
                fields["language"] = config["language"]
            body, content_type = multipart(audio, fields)
            headers = {"Content-Type": content_type}
            if config["api_key"]:
                headers["Authorization"] = "Bearer " + config["api_key"]
            result = self._request(url, body, headers)
            text = checked_text(result.get("text"))
            observed = result.get("language", config["language"])
            language = (
                observed
                if isinstance(observed, str) and observed in LANGUAGES
                else config["language"]
            )
            return text, language
        url = config["azure_endpoint"] or (
            "https://" + config["azure_region"] + ".stt.speech.microsoft.com"
            "/speech/recognition/conversation/cognitiveservices/v1"
        )
        url, _ = endpoint(url)
        url += "?" + urllib.parse.urlencode(
            {"language": config["azure_language"], "format": "simple"}
        )
        result = self._request(
            url,
            audio,
            {
                "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
                "Ocp-Apim-Subscription-Key": config["azure_api_key"],
            },
        )
        status = result.get("RecognitionStatus")
        language = AZURE_LANGUAGES[config["azure_language"]]
        if status in {"NoMatch", "InitialSilenceTimeout", "BabbleTimeout"}:
            return "", language
        if status != "Success":
            raise ASRError("asr_api_response_invalid")
        return checked_text(result.get("DisplayText")), language
