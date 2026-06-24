from proxy_pool import normalize_proxy, parse_proxy_lines


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
