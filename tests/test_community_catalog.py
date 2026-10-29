import copy
import io
import json
import urllib.error
from datetime import UTC, datetime, timedelta

import pytest

from tools.catalog_network import REPOSITORY_ID, SCHEMAS, CatalogUnavailable, GitHub
from tools.catalog_refresh import collect, discover
from tools.catalog_snapshot import DATA_FILES, build, downloads, validate_snapshot
from tools.publish_catalog import publish


def entry(package_id="example.gain", state="available"):
    declaration = json.loads(
        (SCHEMAS.parent / "modules/tts/obliviousmic.module.json").read_text("utf-8")
    )
    declaration.update(
        module_id=package_id, module_version="1.0.0", repository="https://github.com/Alice/gain"
    )
    declaration["distribution"] = {
        "repository_id": 7,
        "asset_id": 9,
        "release_id": 3,
        "release": "v1.0.0",
        "url": "https://github.com/Alice/gain/releases/download/v1.0.0/gain.zip",
        "bytes": 100,
        "sha256": "b" * 64,
    }
    return {
        "package_type": "module",
        "package_id": package_id,
        "version": "1.0.0",
        "source": {
            "repository_id": 7,
            "repository": "Alice/gain",
            "owner": "Alice",
            "owner_id": 8,
            "commit": "a" * 40,
            "declaration_path": "obliviousmic.module.json",
            "release_id": 3,
            "asset_id": 9,
        },
        "declaration": declaration,
        "downloads": {"asset_id": 9, "count": 200},
        "check_state": state,
    }


def test_download_history_is_a_baseline_then_observed_increments_across_asset_versions():
    a = {"9": {"count": 100, "created_at": "2026-01-01", "digest": "a"}}
    first = downloads({}, a, "2026-10-01T00:00:00+00:00")
    assert first["observed_download_delta"] == 0 and first["coverage_days"] == 0
    second = downloads(first, {"9": {**a["9"], "count": 115}}, "2026-10-02T00:00:00+00:00")
    assert second["observed_download_delta"] == 15
    third = downloads(
        second,
        {"9": {**a["9"], "count": 120}, "10": {"count": 50, "created_at": "new", "digest": "b"}},
        "2026-10-03T00:00:00+00:00",
    )
    assert third["observed_download_delta"] == 20
    fourth = downloads(
        third,
        {"9": {**a["9"], "count": 121}, "10": {"count": 60, "created_at": "new", "digest": "b"}},
        "2026-10-04T00:00:00+00:00",
    )
    assert fourth["observed_download_delta"] == 31
    reset = downloads(fourth, {"9": {**a["9"], "count": 1}}, "2026-10-05T00:00:00+00:00")
    assert reset["counter_resets"] and reset["observed_download_delta"] == 31


def test_complete_snapshots_are_atomic_consistent_and_short_failures_remain_recommendable():
    item = entry()
    snapshot = build(
        [
            item,
            entry("example.stale", "stale"),
            entry("example.gone", "unavailable"),
        ],
        {},
        now="2026-10-30T00:00:00+00:00",
        snapshot_id="fixture",
    )
    assert len(snapshot["popular-modules.json"]["entries"]) == 2
    validate_snapshot(snapshot)
    changed = copy.deepcopy(snapshot)
    changed["status.json"]["complete"] = False
    with pytest.raises(ValueError, match="incomplete"):
        validate_snapshot(changed)
    changed = copy.deepcopy(snapshot)
    changed["community-modules.json"]["entries"] = []
    with pytest.raises(ValueError, match="digest"):
        validate_snapshot(changed)
    changed = copy.deepcopy(snapshot)
    changed["popular-modules.json"]["snapshot_id"] = "mixed"
    with pytest.raises(ValueError, match="mixed"):
        validate_snapshot(changed)


def test_search_partitions_beyond_github_ceiling_and_never_accepts_incomplete_pages():
    class Client:
        def search(self, kind, page=1, created=None):
            start, end = created.split("..")
            width = datetime.fromisoformat(end.replace("Z", "+00:00")) - datetime.fromisoformat(
                start.replace("Z", "+00:00")
            )
            return (
                {"total_count": 1001, "items": []}
                if width > timedelta(days=1)
                else {"total_count": 1, "items": [{"id": start}]}
            )

    results = discover(
        Client(), "module", datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 3, tzinfo=UTC)
    )
    assert len(results) == 2

    class Incomplete:
        def search(self, *_a, **_k):
            return {"total_count": 101, "items": []}

    with pytest.raises(CatalogUnavailable, match="incomplete"):
        discover(
            Incomplete(),
            "module",
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 2, tzinfo=UTC),
        )


def test_api_tokens_never_follow_external_redirects_and_user_terms_cannot_inject_qualifiers():
    seen = []

    def opener(request, **_):
        seen.append(request)
        response = io.BytesIO(
            json.dumps({"total_count": 0, "items": [], "incomplete_results": False}).encode()
        )
        response.geturl = lambda: request.full_url
        return response

    client = GitHub("fixture-token", opener=opener)
    client.search("module", "hello topic:evil OR user:attacker", per_page=20)
    from urllib.parse import parse_qs, urlparse

    query = parse_qs(urlparse(seen[0].full_url).query)["q"][0]
    assert "topic:evil" not in query and "user:attacker" not in query
    assert "topic:obliviousmic-module" in query

    def redirect(request, **_):
        raise urllib.error.HTTPError(
            request.full_url, 302, "redirect", {"Location": "https://evil.invalid/token"}, None
        )

    with pytest.raises(CatalogUnavailable):
        GitHub("fixture-token", opener=redirect).request("/repos/Alice/gain")

    def incomplete(request, **_):
        response = io.BytesIO(b'{"incomplete_results":true,"total_count":0,"items":[]}')
        response.geturl = lambda: request.full_url
        return response

    with pytest.raises(CatalogUnavailable, match="incomplete"):
        GitHub(opener=incomplete).search("theme")


def test_collector_preserves_last_valid_entries_on_transient_failure_but_aborts_rate_limits():
    item = entry()
    previous = build([item], {}, now="2026-10-29T00:00:00+00:00", snapshot_id="old")

    class Client:
        def search(self, *_a, **_k):
            return {"total_count": 0, "items": []}

        def repository(self, name):
            raise CatalogUnavailable("github_unavailable")

    now = datetime(2026, 10, 30, tzinfo=UTC)
    result = collect(Client(), previous, now=now)
    assert result["community-modules.json"]["entries"][0]["check_state"] == "stale"
    assert len(result["popular-modules.json"]["entries"]) == 1

    class Limited(Client):
        def repository(self, name):
            raise CatalogUnavailable("github_rate_limited")

    with pytest.raises(CatalogUnavailable, match="rate_limited"):
        collect(Limited(), previous, now=now)
    assert previous["community-modules.json"]["entries"][0]["check_state"] == "available"


def test_publisher_writes_only_generated_catalog_branch_with_no_force_and_checks_before_writes():
    snapshot = build([], {}, now="2026-10-30T00:00:00+00:00", snapshot_id="fixture")

    class Writer:
        def __init__(self):
            self.writes = []

        def repository(self, name):
            return {"id": REPOSITORY_ID}

        def request(self, path, params=None):
            if "ref/heads/" in path:
                return {"object": {"sha": "a" * 40}}
            return {"truncated": False, "tree": [{"path": p, "type": "blob"} for p in DATA_FILES]}

        def write(self, path, data, **kwargs):
            self.writes.append((path, data, kwargs))
            return {"sha": "b" * 40}

    writer = Writer()
    publish(writer, snapshot)
    assert len(writer.writes) == 3
    assert {e["path"] for e in writer.writes[0][1]["tree"]} == DATA_FILES
    assert writer.writes[-1][1]["force"] is False
    assert writer.writes[-1][0].endswith("refs/heads/catalog-data")
    invalid = copy.deepcopy(snapshot)
    invalid["status.json"]["complete"] = False
    writer = Writer()
    with pytest.raises(ValueError):
        publish(writer, invalid)
    assert writer.writes == []
