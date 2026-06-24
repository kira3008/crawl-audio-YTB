"""proxy_pool.py — fetch, validate, xoay proxy free/custom cho yt-dlp."""

import json
import re
import threading
from pathlib import Path

import requests

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


DEFAULT_SOURCES = [
    "https://api.proxyscrape.com/v4/free-proxy-list/get?request=display_proxies&protocol=http&proxy_format=ipport&format=text",
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
    "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
]


class ProxyPool:
    def __init__(self, sources=None, custom_file="proxies.txt",
                 cache_file="proxy_cache.json",
                 validate_url="https://www.youtube.com", timeout=5.0):
        self.sources = DEFAULT_SOURCES if sources is None else list(sources)
        self.custom_file = custom_file
        self.cache_file = cache_file
        self.validate_url = validate_url
        self.timeout = timeout
        self._alive: list[str] = []
        self._idx = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def set_alive(self, proxies: list[str]) -> None:
        with self._lock:
            self._alive = list(proxies)
            self._idx = 0

    def alive_count(self) -> int:
        with self._lock:
            return len(self._alive)

    def get_proxy(self) -> str | None:
        with self._lock:
            if not self._alive:
                return None
            proxy = self._alive[self._idx % len(self._alive)]
            self._idx += 1
            return proxy

    def mark_bad(self, proxy: str) -> None:
        with self._lock:
            if proxy in self._alive:
                self._alive.remove(proxy)
                self._idx = 0

    def save_cache(self) -> None:
        if not self.cache_file:
            return
        try:
            with self._lock:
                data = list(self._alive)
            Path(self.cache_file).write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def load_cache(self) -> int:
        if not self.cache_file:
            return 0
        try:
            cf = Path(self.cache_file)
            if not cf.exists():
                return 0
            data = json.loads(cf.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self.set_alive([str(x) for x in data])
        except Exception:
            return 0
        return self.alive_count()

    def refresh(self) -> int:
        alive = self._validate(self._fetch_raw())
        self.set_alive(alive)
        self.save_cache()
        return len(alive)

    def _fetch_raw(self) -> list[str]:
        lines: list[str] = []
        if self.custom_file:
            try:
                cf = Path(self.custom_file)
                if cf.exists():
                    lines.extend(cf.read_text(encoding="utf-8", errors="replace").splitlines())
            except Exception:
                pass
        for src in self.sources:
            try:
                r = requests.get(src, timeout=self.timeout)
                if r.status_code == 200:
                    lines.extend(r.text.splitlines())
            except Exception:
                continue
        return parse_proxy_lines("\n".join(lines))

    def _validate(self, proxies: list[str]) -> list[str]:
        alive: list[str] = []
        for px in proxies:
            try:
                r = requests.get(self.validate_url,
                                 proxies={"http": px, "https": px},
                                 timeout=self.timeout)
                if r.status_code == 200:
                    alive.append(px)
            except Exception:
                continue
        return alive

    def start_background(self, interval_sec: int = 600) -> None:
        self._stop.clear()

        def _loop():
            self.refresh()
            while not self._stop.wait(interval_sec):
                self.refresh()

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop_background(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
