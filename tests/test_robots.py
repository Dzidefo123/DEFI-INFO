"""robots.txt gate tests. httpx.get is monkeypatched — no network."""

import httpx
import pytest

from src.ingest import robots


class _Resp:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


@pytest.fixture(autouse=True)
def _clear_cache():
    robots._rules.cache_clear()
    yield
    robots._rules.cache_clear()


def _patch(monkeypatch, resp_or_exc):
    def fake_get(url, **kwargs):
        if isinstance(resp_or_exc, Exception):
            raise resp_or_exc
        return resp_or_exc

    monkeypatch.setattr(robots.httpx, "get", fake_get)


def test_disallow_rule_blocks_matching_path(monkeypatch):
    _patch(monkeypatch, _Resp(200, "User-agent: *\nDisallow: /private\n"))
    assert robots.allowed("https://h.io/public/page")
    assert not robots.allowed("https://h.io/private/secret")


def test_missing_robots_allows_everything(monkeypatch):
    _patch(monkeypatch, _Resp(404, "Not Found"))
    assert robots.allowed("https://h.io/anything")


def test_server_error_backs_off(monkeypatch):
    _patch(monkeypatch, _Resp(503))
    assert not robots.allowed("https://h.io/anything")


def test_network_error_backs_off(monkeypatch):
    _patch(monkeypatch, httpx.ConnectError("boom"))
    assert not robots.allowed("https://h.io/anything")


def test_non_http_url_rejected(monkeypatch):
    _patch(monkeypatch, _Resp(200, ""))
    assert not robots.allowed("not-a-url")
