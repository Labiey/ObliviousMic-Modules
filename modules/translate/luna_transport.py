# SPDX-License-Identifier: GPL-3.0-only
"""Bounded in-memory HTTP session implementing Luna leaf engines' request surface."""

import http.cookiejar
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib

from obliviousmic.sdk import PublicModuleError

MAX_BODY = 2 * 1024 * 1024


class Response:
    def __init__(self, status, headers, body):
        self.status_code, self.headers, self.content = status, headers, body

    @property
    def text(self):
        return self.content.decode("utf-8", errors="replace")

    def json(self):
        return json.loads(self.content)

    def __repr__(self):
        return f"<TranslationResponse status={self.status_code}>"


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Session:
    def __init__(self, hosts, timeout, opener=None):
        self.hosts, self.timeout = set(hosts), timeout
        self.cookies = http.cookiejar.CookieJar()
        self.headers = {"User-Agent": "Mozilla/5.0 ObliviousMic-Translate/1.1"}
        self.open = (
            opener
            or urllib.request.build_opener(
                NoRedirects(), urllib.request.HTTPCookieProcessor(self.cookies)
            ).open
        )
        self.begin()

    def begin(self):
        self.deadline = time.monotonic() + self.timeout
        self.calls = 0
        self.error = None

    def checked_url(self, url):
        parsed = urllib.parse.urlsplit(url)
        if (
            parsed.hostname not in self.hosts
            or parsed.username
            or parsed.password
            or parsed.fragment
            or parsed.port not in {None, 80, 443}
            or parsed.scheme not in {"http", "https"}
        ):
            raise PublicModuleError("translate_endpoint_invalid")
        # Two upstream API leaves use legacy HTTP URLs. Never transmit keys in cleartext.
        return urllib.parse.urlunsplit(parsed._replace(scheme="https"))

    def request(
        self,
        method,
        url,
        *,
        params=None,
        data=None,
        json=None,
        headers=None,
        cookies=None,
        allow_redirects=True,
    ):
        self.calls += 1
        if self.calls > 16 or time.monotonic() >= self.deadline:
            self.error = PublicModuleError("translate_request_timeout")
            raise self.error
        url = self.checked_url(url)
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params, doseq=True)
        merged = {**self.headers, **(headers or {}), "Accept-Encoding": "identity"}
        if cookies:
            merged["Cookie"] = "; ".join(f"{k}={v}" for k, v in cookies.items())
        if json is not None:
            payload = __import__("json").dumps(json, ensure_ascii=False).encode("utf-8")
            merged.setdefault("Content-Type", "application/json")
        elif isinstance(data, dict):
            payload = urllib.parse.urlencode(data, doseq=True).encode("utf-8")
            merged.setdefault("Content-Type", "application/x-www-form-urlencoded")
        elif isinstance(data, str):
            # This upstream SDK deliberately passes UTF-8 bytes decoded as Latin-1.
            payload = data.encode("latin1" if "open.volcengineapi.com" in url else "utf-8")
        else:
            payload = data
        request = urllib.request.Request(url, data=payload, headers=merged, method=method)
        try:
            try:
                response = self.open(request, timeout=max(0.01, self.deadline - time.monotonic()))
            except urllib.error.HTTPError as exc:
                if exc.code not in {301, 302, 303, 307, 308}:
                    code = exc.code
                    exc.close()
                    raise PublicModuleError("translate_provider_http_" + str(code)) from None
                response = exc
            with response:
                status = getattr(response, "status", getattr(response, "code", 200))
                response_headers = response.headers
                body = response.read(MAX_BODY + 1)
                if len(body) > MAX_BODY:
                    raise PublicModuleError("translate_response_exceeds_limit")
            if (
                status in {301, 302, 303, 307, 308}
                and allow_redirects
                and method in {"GET", "HEAD"}
            ):
                redirect = self.checked_url(urllib.parse.urljoin(url, response_headers["Location"]))
                return self.request(method, redirect, headers=headers)
            encoding = response_headers.get("Content-Encoding", "").lower()
            if encoding in {"gzip", "deflate"}:
                decoder = zlib.decompressobj(31 if encoding == "gzip" else 15)
                body = decoder.decompress(body, MAX_BODY + 1)
                if len(body) > MAX_BODY or decoder.unconsumed_tail:
                    raise PublicModuleError("translate_response_exceeds_limit")
            if time.monotonic() >= self.deadline:
                raise PublicModuleError("translate_request_timeout")
            if len(self.cookies) > 256:
                self.cookies.clear()
            return Response(status, response_headers, body)
        except PublicModuleError as exc:
            self.error = exc
            raise
        except (OSError, TimeoutError, ValueError, zlib.error):
            self.error = PublicModuleError("translate_service_unavailable")
            raise self.error from None

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    def options(self, url, **kwargs):
        return self.request("OPTIONS", url, **kwargs)
