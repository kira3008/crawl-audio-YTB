import proxy_pool as pp
from proxy_pool import normalize_proxy, parse_proxy_lines, ProxyPool


def _pool_with(proxies):
    p = ProxyPool(sources=[], custom_file=None, cache_file=None)
    p.set_alive(proxies)
    return p


def test_get_proxy_round_robin():
    p = _pool_with(["http://a:1", "http://b:2", "http://c:3"])
    assert [p.get_proxy() for _ in range(4)] == [
        "http://a:1", "http://b:2", "http://c:3", "http://a:1",
    ]


def test_get_proxy_empty_returns_none():
    p = _pool_with([])
    assert p.get_proxy() is None


def test_mark_bad_removes():
    p = _pool_with(["http://a:1", "http://b:2"])
    p.mark_bad("http://a:1")
    assert p.alive_count() == 1
    assert p.get_proxy() == "http://b:2"


def test_mark_bad_unknown_noop():
    p = _pool_with(["http://a:1"])
    p.mark_bad("http://x:9")
    assert p.alive_count() == 1


def test_normalize_plain_ipport():
    assert normalize_proxy("1.2.3.4:8080") == "http://1.2.3.4:8080"


def test_normalize_keeps_scheme():
    assert normalize_proxy("socks5://1.2.3.4:1080") == "socks5://1.2.3.4:1080"


def test_normalize_strips_whitespace():
    assert normalize_proxy("  1.2.3.4:8080  ") == "http://1.2.3.4:8080"


def test_normalize_invalid_returns_none():
    assert normalize_proxy("") is None
    assert normalize_proxy("not-a-proxy") is None
    assert normalize_proxy("1.2.3.4") is None


def test_parse_lines_dedup_and_order():
    text = "1.2.3.4:8080\n# comment\n\n5.6.7.8:3128\n1.2.3.4:8080\n"
    assert parse_proxy_lines(text) == ["http://1.2.3.4:8080", "http://5.6.7.8:3128"]


def test_fetch_raw_merges_custom_first(tmp_path, monkeypatch):
    custom = tmp_path / "proxies.txt"
    custom.write_text("9.9.9.9:9999\n", encoding="utf-8")

    class FakeResp:
        status_code = 200
        text = "1.2.3.4:8080\n5.6.7.8:3128\n"

    monkeypatch.setattr(pp.requests, "get", lambda *a, **k: FakeResp())
    p = pp.ProxyPool(sources=["http://src"], custom_file=str(custom), cache_file=None)
    raw = p._fetch_raw()
    assert raw[0] == "http://9.9.9.9:9999"           # custom truoc
    assert "http://1.2.3.4:8080" in raw
    assert "http://5.6.7.8:3128" in raw


def test_validate_keeps_only_200(monkeypatch):
    def fake_get(url, proxies=None, timeout=None, **k):
        class R:
            status_code = 200 if proxies["https"] == "http://good:1" else 500
        if proxies["https"] == "http://err:3":
            raise pp.requests.RequestException("boom")
        return R()

    monkeypatch.setattr(pp.requests, "get", fake_get)
    p = pp.ProxyPool(sources=[], custom_file=None, cache_file=None)
    alive = p._validate(["http://good:1", "http://bad:2", "http://err:3"])
    assert alive == ["http://good:1"]
