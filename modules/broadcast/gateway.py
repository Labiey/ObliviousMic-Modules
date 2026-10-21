# SPDX-License-Identifier: Apache-2.0
"""Verified local Broadcast MCP connection; never retry mutations or resume its queue."""

import base64
import json
import os
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path

MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class BroadcastError(ValueError):
    pass


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirect rejected", headers, fp)


def discover_port(configuration=None):
    path = (
        Path(configuration)
        if configuration
        else Path(os.environ.get("APPDATA", "")) / "nvidia-broadcast/gateway.json"
    )
    try:
        if path.stat().st_size > 4096:
            raise ValueError()
        data = json.loads(base64.b64decode(path.read_bytes(), validate=True))
        port = data["port"]
        if type(port) is not int or not 18100 <= port <= 18109:
            raise ValueError()
        return port
    except FileNotFoundError:
        return 18100
    except (OSError, ValueError, KeyError, TypeError):
        raise BroadcastError("broadcast_gateway_configuration_invalid") from None


class Gateway:
    def __init__(self, port=None, *, opener=None, timeout=15):
        port = discover_port() if port is None else port
        if type(port) is not int or not 18100 <= port <= 18109:
            raise BroadcastError("broadcast_gateway_port_invalid")
        self.url = f"http://127.0.0.1:{port}/gateway"
        self.port = port
        self.open = (
            opener
            or urllib.request.build_opener(NoRedirects(), urllib.request.ProxyHandler({})).open
        )
        self.timeout = timeout
        self.sequence = 0
        self.lock = threading.RLock()
        self.session = None
        self.verified = False
        info = self.rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "ObliviousMic", "version": "1"},
            },
        )
        if (
            not isinstance(info.get("serverInfo"), dict)
            or info["serverInfo"].get("name") != "nvidia-broadcast"
        ):
            raise BroadcastError("broadcast_gateway_identity_mismatch")
        version = info["serverInfo"].get("version", "")
        self.version = version[:64] if isinstance(version, str) else ""
        self.verified = True

    def rpc(self, method, params, *, timeout=None):
        with self.lock:
            self.sequence += 1
            sequence = self.sequence
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            }
            if self.session:
                headers["Mcp-Session-Id"] = self.session
            request = urllib.request.Request(
                self.url,
                json.dumps(
                    {"jsonrpc": "2.0", "id": sequence, "method": method, "params": params},
                    allow_nan=False,
                ).encode("utf-8"),
                headers,
            )
            try:
                with self.open(
                    request, timeout=self.timeout if timeout is None else timeout
                ) as response:
                    # The Broadcast gateway returns finite JSON replies, not an SSE subscription.
                    if response.headers.get_content_type() != "application/json":
                        raise BroadcastError("broadcast_gateway_response_invalid")
                    data = response.read(MAX_RESPONSE_BYTES + 1)
                    if len(data) > MAX_RESPONSE_BYTES:
                        raise BroadcastError("broadcast_gateway_response_exceeds_limit")
                    session = response.headers.get("Mcp-Session-Id")
                result = json.loads(data)
                if (
                    not isinstance(result, dict)
                    or result.get("jsonrpc") != "2.0"
                    or type(result.get("id")) is not int
                    or result.get("id") != sequence
                    or result.get("error")
                    or not isinstance(result.get("result"), dict)
                ):
                    raise BroadcastError("broadcast_gateway_response_invalid")
                if session:
                    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,256}", session):
                        raise BroadcastError("broadcast_gateway_session_invalid")
                    self.session = session
                return result["result"]
            except urllib.error.HTTPError as exc:
                exc.close()
                raise BroadcastError("broadcast_gateway_http_" + str(exc.code)) from None
            except (urllib.error.URLError, OSError, TimeoutError):
                raise BroadcastError("broadcast_gateway_unavailable_or_timeout") from None
            except (json.JSONDecodeError, UnicodeError):
                raise BroadcastError("broadcast_gateway_response_invalid") from None

    def call(self, name, *, _timeout=None, **arguments):
        if not self.verified:
            raise BroadcastError("broadcast_gateway_identity_required")
        result = self.rpc("tools/call", {"name": name, "arguments": arguments}, timeout=_timeout)
        data = result.get("structuredContent")
        if not isinstance(data, dict):
            try:
                data = json.loads(result["content"][0]["text"])
            except (KeyError, IndexError, TypeError, ValueError):
                raise BroadcastError("broadcast_tool_response_invalid") from None
        if not isinstance(data, dict):
            raise BroadcastError("broadcast_tool_response_invalid")
        if result.get("isError") or data.get("error"):
            error = data.get("error", {})
            code = error.get("code", "failed") if isinstance(error, dict) else "failed"
            if not isinstance(code, str) or not re.fullmatch(r"[a-zA-Z0-9_]{1,64}", code):
                code = "failed"
            raise BroadcastError("broadcast_tool_" + code)
        return data


def audio_effects(descriptors, mode="noise_removal", strength=0.8, mic_profile="default"):
    choices = {
        "noise_removal": ["noise_removal"],
        "room_echo_removal": ["room_echo_removal"],
        "noise_and_echo": ["noise_removal", "room_echo_removal"],
        "studio_voice": ["studio_voice"],
    }
    if (
        mode not in choices
        or isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not 0 <= strength <= 1
    ):
        raise BroadcastError("broadcast_effect_settings_invalid")
    available = {
        item["id"]: item
        for item in descriptors.get("effects", [])
        if isinstance(item, dict) and "id" in item
    }
    effects = []
    for effect_id in choices[mode]:
        item = available.get(effect_id)
        if not item or item.get("section") != "microphone" or item.get("accessible") is not True:
            raise BroadcastError("broadcast_effect_unavailable")
        if effect_id == "studio_voice":
            profiles = next(
                (
                    p.get("enum", [])
                    for p in item.get("params", [])
                    if p.get("name") == "micProfile"
                ),
                [],
            )
            if mic_profile not in profiles:
                raise BroadcastError("broadcast_voice_profile_unavailable")
            params = {"micProfile": mic_profile}
        else:
            params = {"strength": strength}
        effects.append({"effectId": effect_id, "params": params})
    return effects
