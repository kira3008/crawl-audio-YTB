"""proxy_pool.py — fetch, validate, xoay proxy free/custom cho yt-dlp."""

import re

_IPPORT_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}:\d{2,5}$")
_SCHEME_RE = re.compile(r"^[a-z0-9]+://", re.IGNORECASE)


def normalize_proxy(raw: str) -> str | None:
    if not raw:
        return None
    s = raw.strip()
    if not s or s.startswith("#"):
        return None
    if _SCHEME_RE.match(s):
        return s
    if _IPPORT_RE.match(s):
        return f"http://{s}"
    return None


def parse_proxy_lines(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        p = normalize_proxy(line)
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out
