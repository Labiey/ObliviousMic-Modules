# SPDX-License-Identifier: Apache-2.0
"""Bounded REST clients, with no credential-bearing redirects or automatic retries."""

import io
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import soundfile as sf
import soxr
from obliviousmic.sdk import PublicModuleError

MAX_AUDIO_BYTES = 16 * 1024 * 1024


class TTSError(PublicModuleError):
    pass


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirect rejected", headers, fp)


def language_for(text, requested="auto", observed="auto"):
    for language in (requested, observed):
        if language in {"zh", "en", "ja"}:
            return language
    if any("\u3040" <= c <= "\u30ff" for c in text):
        return "ja"
    return "zh" if any("\u4e00" <= c <= "\u9fff" for c in text) else "en"


def validate_config(config):
    provider = config["provider"]
    if provider == "qwen" and not config["qwen_api_key"].strip():
        raise TTSError("tts_qwen_api_key_required")
    if provider == "qwen":
        if config["qwen_region"] not in {"beijing", "singapore"}:
            raise TTSError("tts_qwen_region_invalid")
        if not config["qwen_model"].strip():
            raise TTSError("tts_qwen_model_required")
        workspace = config.get("qwen_workspace_id", "")
        if workspace and not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", workspace):
            raise TTSError("tts_qwen_workspace_invalid")
    if provider == "azure":
        if not config["azure_api_key"].strip():
            raise TTSError("tts_azure_api_key_required")
        if not re.fullmatch(r"[a-z][a-z0-9-]{1,40}", config["azure_region"]):
            raise TTSError("tts_azure_region_invalid")
        for field, maximum in (("azure_rate", 2.0), ("azure_pitch", 2.0)):
            value = config.get(field, 1.0)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0.5 <= value <= maximum
            ):
                raise TTSError("tts_azure_prosody_invalid")
    if provider not in {"qwen", "azure"}:
        raise TTSError("tts_provider_unknown")
    for key in ("qwen_api_key", "azure_api_key"):
        if any(ord(c) < 32 or ord(c) > 126 for c in config[key]):
            raise TTSError("tts_api_key_contains_invalid_characters")


class Client:
    def __init__(self, config, opener=None, websocket_connect=None):
        validate_config(config)
        self.config = dict(config)
        self.open = opener or urllib.request.build_opener(NoRedirects()).open
        self.websocket_connect = websocket_connect

    def _request(self, url, deadline, *, payload=None, headers=None, limit=MAX_AUDIO_BYTES):
        remaining = min(self.config["request_timeout_seconds"], deadline - time.monotonic())
        if remaining <= 0:
            raise TTSError("tts_request_timeout")
        request = urllib.request.Request(
            url, data=payload, headers={"User-Agent": "ObliviousMic-TTS/1", **(headers or {})}
        )
        try:
            with self.open(request, timeout=remaining) as response:
                data = response.read(limit + 1)
                if len(data) > limit:
                    raise TTSError("tts_response_exceeds_limit")
                return data
        except urllib.error.HTTPError as exc:
            # Provider responses can contain prompts, signed URLs or credentials.
            code = exc.code
            exc.close()
            raise TTSError("tts_provider_http_" + str(code)) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise TTSError("tts_provider_unavailable_or_timeout") from None

    def synthesize(self, text, observed_language="auto"):
        config = self.config
        if not text.strip() or len(text) > config["maximum_text_characters"]:
            raise TTSError("tts_text_empty_or_too_long")
        language = language_for(text, config["language"], observed_language)
        deadline = time.monotonic() + config["request_timeout_seconds"]
        if config["provider"] == "qwen":
            if config["qwen_model"].startswith("qwen-audio-") and "-tts-" in config["qwen_model"]:
                try:
                    from .qwen_websocket import synthesize
                except ImportError:
                    from qwen_websocket import synthesize

                return synthesize(config, text, language, deadline, self.websocket_connect)
            host = (
                "dashscope.aliyuncs.com"
                if config["qwen_region"] == "beijing"
                else "dashscope-intl.aliyuncs.com"
            )
            payload = {
                "model": config["qwen_model"],
                "input": {
                    "text": text,
                    "voice": config["qwen_voice"] or "Cherry",
                    "language_type": {"zh": "Chinese", "en": "English", "ja": "Japanese"}[language],
                },
            }
            body = self._request(
                "https://" + host + "/api/v1/services/aigc/multimodal-generation/generation",
                deadline,
                payload=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Authorization": "Bearer " + config["qwen_api_key"],
                    "Content-Type": "application/json",
                },
                limit=1024 * 1024,
            )
            try:
                result = json.loads(body)
                url = result["output"]["audio"]["url"]
                parsed = urllib.parse.urlparse(url)
                if (
                    parsed.scheme not in {"http", "https"}
                    or not parsed.hostname
                    or parsed.username
                    or parsed.password
                    or not parsed.hostname.endswith(".aliyuncs.com")
                    or parsed.port not in {None, 80, 443}
                ):
                    raise ValueError()
                # Official samples can return HTTP OSS links; fetch over HTTPS.
                url = urllib.parse.urlunparse(
                    parsed._replace(scheme="https", netloc=parsed.hostname)
                )
            except (ValueError, KeyError, TypeError):
                raise TTSError("tts_qwen_audio_response_invalid") from None
            data = self._request(url, deadline)  # Never forward the API key to audio storage.
        elif config["provider"] == "azure":
            url = (
                config["azure_endpoint"]
                or "https://"
                + config["azure_region"]
                + ".tts.speech.microsoft.com/cognitiveservices/v1"
            )
            parsed = urllib.parse.urlparse(url)
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or not parsed.hostname.endswith(
                    (".tts.speech.microsoft.com", ".cognitiveservices.azure.com")
                )
                or parsed.port not in {None, 443}
                or parsed.fragment
            ):
                raise TTSError("tts_azure_endpoint_invalid")
            locale, default_voice = {
                "zh": ("zh-CN", "zh-CN-XiaoxiaoNeural"),
                "en": ("en-US", "en-US-JennyNeural"),
                "ja": ("ja-JP", "ja-JP-NanamiNeural"),
            }[language]
            speak = ET.Element(
                "speak",
                {
                    "version": "1.0",
                    "xmlns": "http://www.w3.org/2001/10/synthesis",
                    "xml:lang": locale,
                },
            )
            voice = ET.SubElement(speak, "voice", {"name": config["azure_voice"] or default_voice})
            rate, pitch = config.get("azure_rate", 1.0), config.get("azure_pitch", 1.0)
            if rate == 1 and pitch == 1:
                voice.text = text
            else:
                # Azure pitch uses a relative percentage; rate accepts a multiplier.
                prosody = ET.SubElement(
                    voice,
                    "prosody",
                    {
                        "rate": format(rate, ".6g"),
                        "pitch": format((pitch - 1) * 100, "+.6g") + "%",
                    },
                )
                prosody.text = text
            data = self._request(
                url,
                deadline,
                payload=ET.tostring(speak, encoding="utf-8"),
                headers={
                    "Ocp-Apim-Subscription-Key": config["azure_api_key"],
                    "Content-Type": "application/ssml+xml",
                    "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm",
                },
            )
        return self._decode(data)

    def _decode(self, data):
        try:
            with sf.SoundFile(io.BytesIO(data)) as source:
                rate, channels = source.samplerate, source.channels
                if (
                    source.format not in {"WAV", "WAVEX", "FLAC"}
                    or not 8000 <= rate <= 192000
                    or channels not in {1, 2}
                    or not 0 < source.frames <= rate * self.config["maximum_audio_seconds"]
                ):
                    raise ValueError()
                samples = source.read(dtype="float32", always_2d=True)
            if not np.isfinite(samples).all() or np.max(np.abs(samples), initial=0) > 1.0001:
                raise ValueError()
            samples = samples.mean(axis=1, keepdims=True, dtype=np.float32)
            if rate != 24000:
                samples = soxr.resample(samples, rate, 24000, quality="HQ")
            return np.ascontiguousarray(samples, dtype=np.float32)
        except (ValueError, RuntimeError):
            raise TTSError("tts_audio_response_invalid") from None
