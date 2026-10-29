"""Bounded GitHub metadata reads. Never execute repository content or send tokens off GitHub."""

import base64
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

from jsonschema import Draft202012Validator

ENTRY_NAMES = {"module": "obliviousmic.module.json", "theme": "obliviousmic.theme.json"}
REPOSITORY_ID = 1418766723
REPOSITORY = "Labiey/ObliviousMic-Modules"
SCHEMAS = Path(__file__).resolve().parents[1] / "schemas"


class CatalogUnavailable(RuntimeError):
    def __init__(self, code, *, status=None, retry_after=0):
        self.code, self.status, self.retry_after = code, status, retry_after
        super().__init__(code)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class GitHub:
    def __init__(self, token=None, *, opener=None, timeout=8, limit=1200, retries=0, seconds=2400):
        self.token = token
        self.opener = opener or urllib.request.build_opener(NoRedirect()).open
        self.timeout, self.limit, self.calls = timeout, limit, 0
        self.retries, self.deadline = retries, time.monotonic() + seconds

    def request(self, path, params=None, *, size=2 * 1024 * 1024, redirects=0, attempt=0):
        if not path.startswith("/") or "\\" in path or "?" in path or "#" in path:
            raise ValueError("invalid GitHub API path")
        if self.calls >= self.limit or time.monotonic() >= self.deadline:
            raise CatalogUnavailable("github_request_budget")
        self.calls += 1
        url = "https://api.github.com" + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "ObliviousMic-Catalog/1",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        try:
            with self.opener(
                urllib.request.Request(url, headers=headers), timeout=self.timeout
            ) as response:
                if urllib.parse.urlparse(response.geturl()).hostname != "api.github.com":
                    raise CatalogUnavailable("github_redirect_rejected")
                data = response.read(size + 1)
        except urllib.error.HTTPError as exc:
            if exc.code in (301, 302, 307, 308) and redirects < 2:
                target = urllib.parse.urlparse(exc.headers.get("Location", ""))
                if (
                    target.scheme == "https"
                    and target.netloc == "api.github.com"
                    and not target.query
                    and not target.fragment
                ):
                    return self.request(target.path, params, size=size, redirects=redirects + 1)
            wait = min(
                300,
                max(
                    1,
                    int(exc.headers.get("Retry-After", "0") or 0),
                    int(exc.headers.get("X-RateLimit-Reset", "0") or 0) - int(time.time()),
                ),
            )
            code = (
                "github_not_found"
                if exc.code == 404
                else "github_rate_limited"
                if exc.code in (403, 429)
                else "github_unavailable"
            )
            if (
                code == "github_rate_limited"
                and attempt < self.retries
                and time.monotonic() + wait < self.deadline
            ):
                remaining = wait
                while remaining:
                    pause = min(60, remaining)
                    time.sleep(pause)
                    remaining -= pause
                return self.request(path, params, size=size, attempt=attempt + 1)
            raise CatalogUnavailable(code, status=exc.code, retry_after=wait) from None
        except OSError:
            raise CatalogUnavailable("github_unavailable") from None
        if len(data) > size:
            raise CatalogUnavailable("github_response_too_large")
        try:
            return json.loads(data)
        except (ValueError, UnicodeError):
            raise CatalogUnavailable("github_invalid_response") from None

    @staticmethod
    def repository_name(value):
        if not isinstance(value, str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value
        ):
            raise ValueError("invalid GitHub repository")
        return value

    def repository(self, name):
        value = self.request("/repos/" + self.repository_name(name))
        if value.get("private") or value.get("disabled") or value.get("archived"):
            raise CatalogUnavailable("repository_unavailable", status=410)
        self.repository_name(value["full_name"])
        if type(value.get("id")) is not int or type(value.get("owner", {}).get("id")) is not int:
            raise CatalogUnavailable("github_invalid_response")
        if value["full_name"].split("/")[0].casefold() != value["owner"]["login"].casefold():
            raise CatalogUnavailable("github_invalid_response")
        return value

    def commit(self, repo):
        branch = urllib.parse.quote(repo["default_branch"], safe="")
        result = self.request(f"/repos/{repo['full_name']}/commits/{branch}")
        if not re.fullmatch(r"[0-9a-f]{40}", result.get("sha", "")):
            raise CatalogUnavailable("github_invalid_response")
        return result["sha"]

    def document(self, repo, path, commit, *, max_size=1024 * 1024):
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("declarations require a fixed commit")
        parts = PurePosixPath(path).parts
        if (
            not parts
            or any(p in {".", ".."} for p in parts)
            or path.startswith("/")
            or "\\" in path
        ):
            raise ValueError("invalid declaration path")
        result = self.request(
            f"/repos/{repo['full_name']}/contents/" + urllib.parse.quote(path, safe="/"),
            {"ref": commit},
        )
        if (
            not isinstance(result, dict)
            or result.get("type") != "file"
            or not 0 < result.get("size", 0) <= max_size
        ):
            raise ValueError("invalid declaration file")
        if result.get("encoding") != "base64" and re.fullmatch(
            r"[0-9a-f]{40}", result.get("sha", "")
        ):
            result = self.request(
                f"/repos/{repo['full_name']}/git/blobs/{result['sha']}", size=max_size * 2
            )
        if result.get("encoding") != "base64":
            raise ValueError("unsupported declaration encoding")
        data = base64.b64decode(result["content"], validate=False)
        if len(data) != result["size"]:
            raise ValueError("invalid declaration size")
        return json.loads(data)

    def search(self, kind, query="", page=1, *, created=None, per_page=100):
        if kind not in ENTRY_NAMES or not 1 <= page <= 10:
            raise ValueError("invalid discovery request")
        # User input cannot insert qualifiers, OR operators or arbitrary topics.
        tokens = [
            t[:40] for t in re.findall(r"[\w.-]+", query) if t.upper() not in {"AND", "OR", "NOT"}
        ][:4]
        q = " ".join(f'"{t}"' for t in tokens)
        q += f" topic:obliviousmic-{kind} in:name,description"
        if created:
            q += " created:" + created
        result = self.request(
            "/search/repositories", {"q": q.strip(), "page": page, "per_page": per_page}
        )
        if (
            result.get("incomplete_results")
            or type(result.get("total_count")) is not int
            or not isinstance(result.get("items"), list)
        ):
            raise CatalogUnavailable("github_search_incomplete")
        return result

    def candidate(self, repo, kind, *, path=None, commit=None):
        commit = commit or self.commit(repo)
        path = path or ENTRY_NAMES[kind]
        document = self.document(repo, path, commit)
        schema = json.loads((SCHEMAS / f"obliviousmic-{kind}-v1.json").read_text("utf-8"))
        Draft202012Validator(schema).validate(document)
        distribution = document.get("distribution")
        if not distribution:
            raise ValueError("distribution metadata is required")
        if (document.get("repository") or "").rstrip("/").casefold() != (
            "https://github.com/" + repo["full_name"]
        ).casefold():
            raise ValueError("declaration repository does not match verified owner")
        if distribution["repository_id"] != repo["id"]:
            raise ValueError("declaration repository ID mismatch")
        tag = urllib.parse.quote(distribution["release"], safe="")
        release = self.request(f"/repos/{repo['full_name']}/releases/tags/{tag}")
        if release.get("draft") or release.get("id") != distribution["release_id"]:
            raise ValueError("declared release is not available")
        asset = next(
            (a for a in release.get("assets", []) if a.get("id") == distribution["asset_id"]), None
        )
        if (
            not asset
            or asset.get("state") != "uploaded"
            or asset.get("size") != distribution["bytes"]
            or asset.get("browser_download_url") != distribution["url"]
        ):
            raise ValueError("declared release asset mismatch")
        digest = asset.get("digest")
        if digest and digest != "sha256:" + distribution["sha256"]:
            raise ValueError("declared release asset digest mismatch")
        prefix = "https://github.com/" + repo["full_name"] + "/releases/download/"
        if not distribution["url"].startswith(prefix):
            raise ValueError("release URL does not match source")
        package_id = document["module_id" if kind == "module" else "id"]
        version = document["module_version" if kind == "module" else "version"]
        return {
            "package_type": kind,
            "package_id": package_id,
            "version": version,
            "declaration": document,
            "source": {
                "repository_id": repo["id"],
                "repository": repo["full_name"],
                "owner": repo["owner"]["login"],
                "owner_id": repo["owner"]["id"],
                "commit": commit,
                "declaration_path": path,
                "release_id": release["id"],
                "asset_id": asset["id"],
            },
            "downloads": {
                "asset_id": asset["id"],
                "count": asset.get("download_count", 0),
                "created_at": asset.get("created_at"),
                "updated_at": asset.get("updated_at"),
            },
            "check_state": "available",
        }
