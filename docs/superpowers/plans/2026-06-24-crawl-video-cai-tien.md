# Crawl Video — 3 Cải Tiến Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thêm proxy pool xoay IP, backend Groq tùy chọn cho Whisper, và bước lọc nhiễu/chuẩn hóa transcript vào pipeline crawl video.

**Architecture:** Tách 3 tính năng mới thành 3 module độc lập (`proxy_pool.py`, `transcribe_backends.py`, `clean_transcript.py`); `main.py` điều phối; `split_audio.py` sửa nhẹ để ưu tiên `.clean.json`. Logic thuần (parsing, math timestamp, heuristic, chunk) tách thành hàm pure để test bằng pytest; phần I/O (proxy fetch, Groq API, ffmpeg) được mock trong test.

**Tech Stack:** Python 3.10+, yt-dlp, whisperx, groq (SDK), requests, python-dotenv, pytest, ffmpeg.

## Global Constraints

- Python 3.10+ (code dùng `str | None`, `list[dict]` syntax — giữ nguyên).
- Schema transcript dùng chung, KHÔNG đổi: `[{"start": "hh:mm:ss.mmm", "end": "hh:mm:ss.mmm", "text": str, "words": [{"word": str, "start": "hh:mm:ss.mmm", "end": "hh:mm:ss.mmm"}]}]`.
- Tương thích ngược: proxy mode mặc định TẮT; backend mặc định LOCAL; không có `.clean.json` thì `split_audio.py` dùng `.json` như cũ.
- `GROQ_API_KEY` đọc từ env hoặc `.env`. Không có key → ẩn/bỏ qua mọi nhánh Groq, không crash.
- File `.json` transcript gốc KHÔNG bao giờ bị ghi đè bởi bước clean.
- Comment/code/commit message: tiếng Việt-không-dấu hoặc tiếng Anh theo style hiện có; KHÔNG đưa xưng hô anh/em vào code.
- Mọi hàm gọi mạng/API/ffmpeg phải có nhánh fail an toàn (trả về fallback, không raise lên pipeline chính).

**3 phase độc lập** — có thể thực thi & merge riêng:
- Phase 0: Setup test infra + dependencies (làm trước).
- Phase 1: Proxy pool (#1).
- Phase 2: Groq backend (#2).
- Phase 3: Clean transcript (#3).

---

## Phase 0 — Setup

### Task 0: Dependencies + test infrastructure

**Files:**
- Modify: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `pytest.ini`

**Interfaces:**
- Consumes: nothing
- Produces: môi trường pytest chạy được; các package `groq`, `requests`, `python-dotenv` khai báo.

- [ ] **Step 1: Cập nhật `requirements.txt`**

Thêm 3 dòng vào cuối `requirements.txt`:

```
groq>=0.11.0
requests>=2.31.0
python-dotenv>=1.0.0
```

- [ ] **Step 2: Tạo `requirements-dev.txt`**

```
-r requirements.txt
pytest>=8.0.0
```

- [ ] **Step 3: Tạo `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -q
```

- [ ] **Step 4: Tạo `tests/__init__.py`** (file rỗng)

```python
```

- [ ] **Step 5: Tạo `tests/conftest.py`**

```python
import sys
from pathlib import Path

# Cho phep import cac module o thu muc goc du an trong test
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

- [ ] **Step 6: Cài dev deps + verify pytest chạy**

Run:
```bash
pip install -r requirements-dev.txt
pytest
```
Expected: pytest chạy, "no tests ran" (exit code 5) hoặc "collected 0 items" — không lỗi import/config.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt requirements-dev.txt pytest.ini tests/__init__.py tests/conftest.py
git commit -m "chore: setup pytest infra and new dependencies"
```

---

## Phase 1 — Proxy Pool (#1)

### Task 1.1: Chuẩn hóa & parse proxy

**Files:**
- Create: `proxy_pool.py`
- Test: `tests/test_proxy_pool.py`

**Interfaces:**
- Produces:
  - `normalize_proxy(raw: str) -> str | None` — chuẩn hóa 1 dòng proxy thành `"http://host:port"`, trả `None` nếu không hợp lệ.
  - `parse_proxy_lines(text: str) -> list[str]` — parse nhiều dòng, normalize, loại trùng, giữ thứ tự.

- [ ] **Step 1: Viết test thất bại**

```python
# tests/test_proxy_pool.py
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
```

- [ ] **Step 2: Chạy test — phải FAIL**

Run: `pytest tests/test_proxy_pool.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'proxy_pool'`

- [ ] **Step 3: Tạo `proxy_pool.py` với 2 hàm**

```python
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
```

- [ ] **Step 4: Chạy test — phải PASS**

Run: `pytest tests/test_proxy_pool.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add proxy_pool.py tests/test_proxy_pool.py
git commit -m "feat(proxy): add proxy normalize and parse helpers"
```

---

### Task 1.2: ProxyPool core — get/mark_bad/round-robin/fallback

**Files:**
- Modify: `proxy_pool.py`
- Test: `tests/test_proxy_pool.py`

**Interfaces:**
- Consumes: `parse_proxy_lines`
- Produces:
  - class `ProxyPool` với:
    - `__init__(self, sources=None, custom_file="proxies.txt", cache_file="proxy_cache.json", validate_url="https://www.youtube.com", timeout=5.0)`
    - `set_alive(self, proxies: list[str]) -> None` — set danh sách proxy sống (dùng trong test & refresh).
    - `get_proxy(self) -> str | None` — round-robin; `None` nếu rỗng.
    - `mark_bad(self, proxy: str) -> None` — loại proxy khỏi danh sách sống.
    - `alive_count(self) -> int`

- [ ] **Step 1: Viết test thất bại**

```python
# them vao tests/test_proxy_pool.py
from proxy_pool import ProxyPool


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
```

- [ ] **Step 2: Chạy test — phải FAIL**

Run: `pytest tests/test_proxy_pool.py -k "round_robin or empty or mark_bad" -v`
Expected: FAIL — `ImportError: cannot import name 'ProxyPool'`

- [ ] **Step 3: Thêm class `ProxyPool` vào `proxy_pool.py`**

Thêm `import threading` ở đầu file (cạnh `import re`), rồi thêm:

```python
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
```

- [ ] **Step 4: Chạy test — phải PASS**

Run: `pytest tests/test_proxy_pool.py -v`
Expected: PASS (toàn bộ test trong file)

- [ ] **Step 5: Commit**

```bash
git add proxy_pool.py tests/test_proxy_pool.py
git commit -m "feat(proxy): ProxyPool core with round-robin and mark_bad"
```

---

### Task 1.3: Fetch nguồn + custom file + validate (mock network)

**Files:**
- Modify: `proxy_pool.py`
- Test: `tests/test_proxy_pool.py`

**Interfaces:**
- Consumes: `parse_proxy_lines`, `ProxyPool`
- Produces:
  - `ProxyPool._fetch_raw(self) -> list[str]` — gọi `requests.get` cho từng source + đọc `custom_file`, gộp (custom trước), dedupe.
  - `ProxyPool._validate(self, proxies: list[str]) -> list[str]` — giữ proxy mà `requests.get(validate_url, proxies=..., timeout=...)` trả status 200.

- [ ] **Step 1: Viết test thất bại (mock `requests`)**

```python
# them vao tests/test_proxy_pool.py
import proxy_pool as pp


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
```

- [ ] **Step 2: Chạy test — phải FAIL**

Run: `pytest tests/test_proxy_pool.py -k "fetch_raw or validate_keeps" -v`
Expected: FAIL — `AttributeError`/`module 'proxy_pool' has no attribute 'requests'`

- [ ] **Step 3: Thêm `requests` import + 2 method**

Thêm `import requests` ở đầu file. Thêm vào class `ProxyPool`:

```python
    def _fetch_raw(self) -> list[str]:
        lines: list[str] = []
        if self.custom_file:
            try:
                from pathlib import Path
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
```

- [ ] **Step 4: Chạy test — phải PASS**

Run: `pytest tests/test_proxy_pool.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add proxy_pool.py tests/test_proxy_pool.py
git commit -m "feat(proxy): fetch from sources/custom file and validate proxies"
```

---

### Task 1.4: Cache persistence + refresh()

**Files:**
- Modify: `proxy_pool.py`
- Test: `tests/test_proxy_pool.py`

**Interfaces:**
- Consumes: `_fetch_raw`, `_validate`, `set_alive`
- Produces:
  - `ProxyPool.save_cache(self) -> None` — ghi `self._alive` ra `cache_file` (JSON list).
  - `ProxyPool.load_cache(self) -> int` — đọc `cache_file` vào `_alive`, trả số lượng.
  - `ProxyPool.refresh(self) -> int` — `_validate(_fetch_raw())` → `set_alive` → `save_cache`, trả số proxy sống.

- [ ] **Step 1: Viết test thất bại**

```python
# them vao tests/test_proxy_pool.py
import json


def test_cache_save_load_roundtrip(tmp_path):
    cache = tmp_path / "cache.json"
    p = pp.ProxyPool(sources=[], custom_file=None, cache_file=str(cache))
    p.set_alive(["http://a:1", "http://b:2"])
    p.save_cache()
    assert json.loads(cache.read_text(encoding="utf-8")) == ["http://a:1", "http://b:2"]

    p2 = pp.ProxyPool(sources=[], custom_file=None, cache_file=str(cache))
    assert p2.load_cache() == 2
    assert p2.alive_count() == 2


def test_refresh_validates_and_caches(tmp_path, monkeypatch):
    cache = tmp_path / "cache.json"
    p = pp.ProxyPool(sources=[], custom_file=None, cache_file=str(cache))
    monkeypatch.setattr(p, "_fetch_raw", lambda: ["http://a:1", "http://b:2"])
    monkeypatch.setattr(p, "_validate", lambda proxies: ["http://a:1"])
    n = p.refresh()
    assert n == 1
    assert p.alive_count() == 1
    assert json.loads(cache.read_text(encoding="utf-8")) == ["http://a:1"]
```

- [ ] **Step 2: Chạy test — phải FAIL**

Run: `pytest tests/test_proxy_pool.py -k "cache_save or refresh_validates" -v`
Expected: FAIL — `AttributeError: 'ProxyPool' object has no attribute 'save_cache'`

- [ ] **Step 3: Thêm 3 method**

Thêm `import json` và `from pathlib import Path` ở đầu file (nếu chưa có), rồi thêm vào class:

```python
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
```

- [ ] **Step 4: Chạy test — phải PASS**

Run: `pytest tests/test_proxy_pool.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add proxy_pool.py tests/test_proxy_pool.py
git commit -m "feat(proxy): cache persistence and refresh()"
```

---

### Task 1.5: Background refresh thread

**Files:**
- Modify: `proxy_pool.py`
- Test: `tests/test_proxy_pool.py`

**Interfaces:**
- Consumes: `refresh`
- Produces:
  - `ProxyPool.start_background(self, interval_sec=600) -> None` — chạy `refresh()` ngay 1 lần, rồi lặp mỗi `interval_sec` trong thread daemon cho tới khi `stop_background`.
  - `ProxyPool.stop_background(self) -> None`.

- [ ] **Step 1: Viết test thất bại**

```python
# them vao tests/test_proxy_pool.py
import threading as _th


def test_background_runs_initial_refresh(monkeypatch):
    p = pp.ProxyPool(sources=[], custom_file=None, cache_file=None)
    called = _th.Event()

    def fake_refresh():
        p.set_alive(["http://a:1"])
        called.set()
        return 1

    monkeypatch.setattr(p, "refresh", fake_refresh)
    p.start_background(interval_sec=999)
    assert called.wait(timeout=3.0)      # refresh lan dau chay ngay
    p.stop_background()
    assert p.alive_count() == 1
```

- [ ] **Step 2: Chạy test — phải FAIL**

Run: `pytest tests/test_proxy_pool.py -k background -v`
Expected: FAIL — `AttributeError: ... 'start_background'`

- [ ] **Step 3: Thêm 2 method**

```python
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
```

- [ ] **Step 4: Chạy test — phải PASS**

Run: `pytest tests/test_proxy_pool.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add proxy_pool.py tests/test_proxy_pool.py
git commit -m "feat(proxy): background refresh thread"
```

---

### Task 1.6: Retry-with-rotation helper + tích hợp main.py

**Files:**
- Create: `tests/test_download_retry.py`
- Modify: `main.py` (hàm `download_one`, `search_videos`, `main` — thêm tham số `pool` và menu)

**Interfaces:**
- Consumes: `ProxyPool`
- Produces:
  - `main.download_with_rotation(video, output_dir, ffmpeg_dir, progress, task_id, pool, max_retries=3) -> bool` — gọi `_download_attempt(...)` (tách từ thân `download_one` hiện tại, nhận thêm `proxy: str | None`); nếu fail và còn proxy thì `pool.mark_bad` + lấy proxy mới retry; pool=None ⇒ chạy 1 lần không proxy.

- [ ] **Step 1: Viết test thất bại cho logic rotation**

```python
# tests/test_download_retry.py
import main
from proxy_pool import ProxyPool


class DummyProgress:
    def update(self, *a, **k):
        pass


def test_rotation_retries_then_succeeds(monkeypatch):
    pool = ProxyPool(sources=[], custom_file=None, cache_file=None)
    pool.set_alive(["http://p1:1", "http://p2:2"])
    tried = []

    def fake_attempt(video, output_dir, ffmpeg_dir, progress, task_id, proxy):
        tried.append(proxy)
        return proxy == "http://p2:2"      # proxy dau that bai, proxy hai thanh cong

    monkeypatch.setattr(main, "_download_attempt", fake_attempt)
    ok = main.download_with_rotation(
        {"url": "u", "title": "t", "id": "i"}, "out", None,
        DummyProgress(), 0, pool, max_retries=3)
    assert ok is True
    assert tried == ["http://p1:1", "http://p2:2"]
    assert pool.alive_count() == 1          # p1 bi mark_bad


def test_rotation_no_pool_runs_once(monkeypatch):
    calls = []

    def fake_attempt(video, output_dir, ffmpeg_dir, progress, task_id, proxy):
        calls.append(proxy)
        return True

    monkeypatch.setattr(main, "_download_attempt", fake_attempt)
    ok = main.download_with_rotation(
        {"url": "u", "title": "t", "id": "i"}, "out", None,
        DummyProgress(), 0, None)
    assert ok is True
    assert calls == [None]
```

- [ ] **Step 2: Chạy test — phải FAIL**

Run: `pytest tests/test_download_retry.py -v`
Expected: FAIL — `AttributeError: module 'main' has no attribute 'download_with_rotation'`

- [ ] **Step 3: Refactor `download_one` → `_download_attempt(proxy)` + thêm `download_with_rotation`**

Trong `main.py`, đổi chữ ký `download_one` thành `_download_attempt` và thêm tham số `proxy`. Tại block `ydl_opts`, sau khi set `ffmpeg_location`, thêm:

```python
    if proxy:
        ydl_opts["proxy"] = proxy
```

Chữ ký mới:

```python
def _download_attempt(
    video: dict,
    output_dir: str,
    ffmpeg_dir: str | None,
    progress,
    task_id: int,
    proxy: str | None = None,
) -> bool:
```

Thêm hàm mới ngay dưới `_download_attempt`:

```python
def download_with_rotation(
    video: dict,
    output_dir: str,
    ffmpeg_dir: str | None,
    progress,
    task_id: int,
    pool=None,
    max_retries: int = 3,
) -> bool:
    if pool is None:
        return _download_attempt(video, output_dir, ffmpeg_dir, progress, task_id, None)

    for _ in range(max_retries):
        proxy = pool.get_proxy()
        ok = _download_attempt(video, output_dir, ffmpeg_dir, progress, task_id, proxy)
        if ok:
            return True
        if proxy:
            pool.mark_bad(proxy)
        else:
            break          # khong con proxy -> dung
    return False
```

- [ ] **Step 4: Chạy test — phải PASS**

Run: `pytest tests/test_download_retry.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Cập nhật `search_videos` nhận proxy + chỗ gọi download trong `main()`**

Trong `search_videos`, thêm tham số `proxy: str | None = None` và chèn vào `cmd` (trước phần `ytsearch...`):

```python
    if proxy:
        cmd += ["--proxy", proxy]
```

Trong `main()`, thay block submit executor từ `download_one` sang `download_with_rotation`:

```python
            future_map = {
                executor.submit(
                    download_with_rotation, v, output_dir, ffmpeg_dir,
                    progress, task_map[v["id"]], pool,
                ): v
                for v in to_download
            }
```

- [ ] **Step 6: Thêm menu bật proxy + khởi tạo pool trong `main()`**

Sau block chọn `max_workers` (questionary), thêm:

```python
    use_proxy = questionary.confirm(
        "Bật proxy pool (xoay IP free để crawl số lượng lớn)?",
        default=False,
    ).ask()
    if use_proxy is None:
        return

    pool = None
    if use_proxy:
        from proxy_pool import ProxyPool
        pool = ProxyPool()
        n_cache = pool.load_cache()
        console.print(f"[dim]Proxy cache: {n_cache} proxy[/dim]")
        with console.status("[bold green]Đang lấy & kiểm tra proxy free…[/bold green]"):
            n = pool.refresh()
        console.print(f"[green]✓ {n} proxy sống[/green]")
        pool.start_background(interval_sec=600)
```

(Lưu ý: `console` đã được tạo trước đó trong `main()`; block này đặt SAU dòng tạo `console`.)

- [ ] **Step 7: Dừng background thread cuối `main()`**

Ngay trước `console.print(Panel(` của summary cuối cùng, thêm:

```python
    if pool is not None:
        pool.stop_background()
```

- [ ] **Step 8: Smoke test import + full proxy suite**

Run:
```bash
python -c "import main; print(main.download_with_rotation.__name__)"
pytest tests/test_proxy_pool.py tests/test_download_retry.py -v
```
Expected: in `download_with_rotation`; toàn bộ test PASS.

- [ ] **Step 9: Commit**

```bash
git add main.py tests/test_download_retry.py
git commit -m "feat(proxy): integrate proxy pool with download rotation and menu"
```

---

## Phase 2 — Groq Backend (#2)

### Task 2.1: Tách backend local sang `transcribe_backends.py`

**Files:**
- Create: `transcribe_backends.py`
- Modify: `main.py` (import + `transcribe_audio` gọi backend)
- Test: `tests/test_transcribe_backends.py`

**Interfaces:**
- Produces:
  - `_sec_to_hms(sec: float) -> str` (move từ main.py).
  - `load_local_model(model_name: str) -> dict` — bundle `{"asr","align","meta","device"}` (logic cũ của `_load_whisper_model`, bỏ phần trả `backend`).
  - `transcribe_local(mp3_path: str, bundle: dict) -> list[dict]` — trả entries theo schema chung (KHÔNG ghi file).

- [ ] **Step 1: Viết test thất bại (chỉ test hàm pure `_sec_to_hms`)**

```python
# tests/test_transcribe_backends.py
from transcribe_backends import _sec_to_hms


def test_sec_to_hms_basic():
    assert _sec_to_hms(0) == "00:00:00.000"
    assert _sec_to_hms(61.5) == "00:01:01.500"
    assert _sec_to_hms(3661.250) == "01:01:01.250"


def test_sec_to_hms_ms_rounding_carry():
    assert _sec_to_hms(0.9996) == "00:00:01.000"
```

- [ ] **Step 2: Chạy test — phải FAIL**

Run: `pytest tests/test_transcribe_backends.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'transcribe_backends'`

- [ ] **Step 3: Tạo `transcribe_backends.py`, move logic local**

Tạo file với nội dung (copy nguyên `_sec_to_hms` từ main.py; gộp `_detect_gpu`, `_load_whisper_model`, phần thân `transcribe_audio` thành `transcribe_local`):

```python
"""transcribe_backends.py — backend transcribe: WhisperX local + Groq API.

Tat ca tra ve schema chung:
  [{"start","end","text","words":[{"word","start","end"}]}]  (hms hh:mm:ss.mmm)
"""

import subprocess
import logging
from pathlib import Path


def _sec_to_hms(sec: float) -> str:
    ms = round((sec % 1) * 1000)
    total = int(sec)
    if ms == 1000:
        ms = 0
        total += 1
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _detect_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False


def load_local_model(model_name: str) -> dict:
    import warnings
    warnings.filterwarnings("ignore", message="torchcodec is not installed")
    import whisperx

    device = "cpu"
    download_root = str(Path(__file__).parent / "models")
    if _detect_gpu():
        try:
            asr_model = whisperx.load_model(model_name, device="cuda", compute_type="int8",
                                            language="vi", download_root=download_root)
            device = "cuda"
        except Exception:
            asr_model = whisperx.load_model(model_name, device="cpu", compute_type="int8",
                                            language="vi", download_root=download_root)
    else:
        asr_model = whisperx.load_model(model_name, device="cpu", compute_type="int8",
                                        language="vi", download_root=download_root)

    align_model, metadata = whisperx.load_align_model(language_code="vi", device=device)
    return {"asr": asr_model, "align": align_model, "meta": metadata, "device": device}


def transcribe_local(mp3_path: str, bundle: dict) -> list[dict]:
    import whisperx
    device = bundle["device"]
    batch_size = 16 if device == "cuda" else 4

    audio = whisperx.load_audio(str(mp3_path))

    result = None
    while batch_size >= 1:
        try:
            import torch
            if device == "cuda":
                torch.cuda.empty_cache()
            result = bundle["asr"].transcribe(audio, batch_size=batch_size, language="vi")
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and batch_size > 1:
                batch_size = batch_size // 2
                logging.warning(f"[transcribe] OOM — giam batch_size xuong {batch_size}")
            else:
                raise
    if result is None:
        return []

    aligned = whisperx.align(
        result["segments"], bundle["align"], bundle["meta"], audio,
        device=device, return_char_alignments=False,
    )

    entries = []
    for seg in aligned["segments"]:
        text = seg.get("text", "").strip()
        if not text:
            continue
        words = [w for w in seg.get("words", []) if "start" in w and "end" in w]
        if words:
            start, end = words[0]["start"], words[-1]["end"]
        else:
            start, end = seg["start"], seg["end"]
        entries.append({
            "start": _sec_to_hms(start),
            "end": _sec_to_hms(end),
            "text": text,
            "words": [
                {"word": w["word"], "start": _sec_to_hms(w["start"]), "end": _sec_to_hms(w["end"])}
                for w in words
            ],
        })
    return entries
```

- [ ] **Step 4: Chạy test — phải PASS**

Run: `pytest tests/test_transcribe_backends.py -v`
Expected: PASS

- [ ] **Step 5: Sửa `main.py` dùng backend mới**

Trong `main.py`:
- Xóa hàm `_detect_gpu`, `_load_whisper_model`, và phần thân nhận dạng trong `transcribe_audio` (giữ phần kiểm tra file + ghi JSON).
- Thêm import đầu file: `from transcribe_backends import load_local_model, transcribe_local`.
- Thay `transcribe_audio` thành:

```python
def transcribe_audio(safe_title: str, output_dir: str, model, backend: str = "local") -> bool:
    import logging
    mp3_path = Path(output_dir) / f"{safe_title}.mp3"
    if not mp3_path.exists():
        logging.warning(f"[transcribe] MP3 not found: {mp3_path}")
        return False
    try:
        if backend == "groq":
            from transcribe_backends import transcribe_groq
            entries = transcribe_groq(str(mp3_path), model, get_ffmpeg_dir())
        else:
            entries = transcribe_local(str(mp3_path), model)
        if not entries:
            logging.warning(f"[transcribe] No segments: {mp3_path}")
            return False
        json_path = Path(output_dir) / f"{safe_title}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logging.error(f"[transcribe] Error {mp3_path}: {e}", exc_info=True)
        return False
```

- Thay chỗ gọi `_load_whisper_model` trong `main()`:

```python
    wmodel = load_local_model(whisper_model)
    device = wmodel["device"]
    backend = "local"
```

(Lưu ý: `transcribe_groq` được tham chiếu ở đây nhưng tạo ở Task 2.5; nếu thực thi Phase 2 theo thứ tự, Task 2.5 hoàn thành trước khi chạy nhánh groq. Import nằm trong nhánh `if backend=="groq"` nên không lỗi khi chưa có.)

- [ ] **Step 6: Smoke test + commit**

Run:
```bash
python -c "import main, transcribe_backends; print('ok')"
pytest tests/test_transcribe_backends.py -v
```
Expected: in `ok`; test PASS.

```bash
git add transcribe_backends.py main.py tests/test_transcribe_backends.py
git commit -m "refactor(transcribe): extract local backend to transcribe_backends.py"
```

---

### Task 2.2: Map Groq verbose_json → schema chung

**Files:**
- Modify: `transcribe_backends.py`
- Test: `tests/test_transcribe_backends.py`

**Interfaces:**
- Consumes: `_sec_to_hms`
- Produces:
  - `groq_response_to_entries(resp: dict, offset_sec: float = 0.0) -> list[dict]` — đọc `resp["segments"]` (mỗi seg có `start`,`end`,`text`, tùy chọn `words`=[{"word"/"text","start","end"}]); cộng `offset_sec`; bỏ segment text rỗng; trả schema chung.

- [ ] **Step 1: Viết test thất bại**

```python
# them vao tests/test_transcribe_backends.py
from transcribe_backends import groq_response_to_entries


def test_groq_map_basic_with_words():
    resp = {"segments": [
        {"start": 1.0, "end": 2.5, "text": " Xin chao ",
         "words": [{"word": "Xin", "start": 1.0, "end": 1.4},
                   {"word": "chao", "start": 1.5, "end": 2.5}]},
        {"start": 3.0, "end": 3.0, "text": "  "},   # rong -> bo
    ]}
    out = groq_response_to_entries(resp)
    assert len(out) == 1
    assert out[0]["start"] == "00:00:01.000"
    assert out[0]["end"] == "00:00:02.500"
    assert out[0]["text"] == "Xin chao"
    assert out[0]["words"][1]["word"] == "chao"


def test_groq_map_offset_and_missing_words():
    resp = {"segments": [{"start": 0.0, "end": 1.0, "text": "A"}]}
    out = groq_response_to_entries(resp, offset_sec=600.0)
    assert out[0]["start"] == "00:10:00.000"
    assert out[0]["words"] == []
```

- [ ] **Step 2: Chạy test — phải FAIL**

Run: `pytest tests/test_transcribe_backends.py -k groq_map -v`
Expected: FAIL — `ImportError: cannot import name 'groq_response_to_entries'`

- [ ] **Step 3: Thêm hàm map**

```python
def groq_response_to_entries(resp: dict, offset_sec: float = 0.0) -> list[dict]:
    entries: list[dict] = []
    for seg in resp.get("segments", []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        words_out = []
        for w in seg.get("words") or []:
            token = w.get("word", w.get("text", ""))
            if "start" in w and "end" in w:
                words_out.append({
                    "word": token,
                    "start": _sec_to_hms(float(w["start"]) + offset_sec),
                    "end": _sec_to_hms(float(w["end"]) + offset_sec),
                })
        entries.append({
            "start": _sec_to_hms(float(seg["start"]) + offset_sec),
            "end": _sec_to_hms(float(seg["end"]) + offset_sec),
            "text": text,
            "words": words_out,
        })
    return entries
```

- [ ] **Step 4: Chạy test — phải PASS**

Run: `pytest tests/test_transcribe_backends.py -k groq_map -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add transcribe_backends.py tests/test_transcribe_backends.py
git commit -m "feat(groq): map verbose_json response to common schema"
```

---

### Task 2.3: Tính cửa sổ chunk cho file dài

**Files:**
- Modify: `transcribe_backends.py`
- Test: `tests/test_transcribe_backends.py`

**Interfaces:**
- Produces:
  - `plan_chunks(duration_sec: float, max_chunk_sec: float = 600.0, overlap_sec: float = 5.0) -> list[tuple[float, float]]` — chia `[0, duration]` thành các cửa sổ dài ≤ `max_chunk_sec`, mỗi cửa sổ (trừ cái đầu) lùi lại `overlap_sec`. `duration<=max_chunk_sec` ⇒ 1 cửa sổ `[0, duration]`.

- [ ] **Step 1: Viết test thất bại**

```python
# them vao tests/test_transcribe_backends.py
from transcribe_backends import plan_chunks


def test_plan_chunks_single_when_short():
    assert plan_chunks(300, max_chunk_sec=600) == [(0.0, 300.0)]


def test_plan_chunks_multiple_with_overlap():
    chunks = plan_chunks(1300, max_chunk_sec=600, overlap_sec=5)
    assert chunks[0] == (0.0, 600.0)
    assert chunks[1][0] == 595.0           # lui lai overlap
    assert chunks[1][1] == 1195.0
    assert chunks[-1][1] == 1300.0         # phu het toi cuoi
```

- [ ] **Step 2: Chạy test — phải FAIL**

Run: `pytest tests/test_transcribe_backends.py -k plan_chunks -v`
Expected: FAIL — `ImportError: cannot import name 'plan_chunks'`

- [ ] **Step 3: Thêm hàm**

```python
def plan_chunks(duration_sec: float, max_chunk_sec: float = 600.0,
                overlap_sec: float = 5.0) -> list[tuple[float, float]]:
    if duration_sec <= max_chunk_sec:
        return [(0.0, float(duration_sec))]
    chunks: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_sec:
        end = min(start + max_chunk_sec, duration_sec)
        chunks.append((start, end))
        if end >= duration_sec:
            break
        start = end - overlap_sec
    return chunks
```

- [ ] **Step 4: Chạy test — phải PASS**

Run: `pytest tests/test_transcribe_backends.py -k plan_chunks -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add transcribe_backends.py tests/test_transcribe_backends.py
git commit -m "feat(groq): plan_chunks for long audio windows"
```

---

### Task 2.4: Merge entries từ nhiều chunk (khử trùng overlap)

**Files:**
- Modify: `transcribe_backends.py`
- Test: `tests/test_transcribe_backends.py`

**Interfaces:**
- Produces:
  - `merge_chunk_entries(entry_lists: list[list[dict]]) -> list[dict]` — gộp các list entries (đã offset về thời gian tuyệt đối), sắp theo `start`, loại entry trùng (cùng `text` và `start` lệch < 1.0s so với entry đã giữ liền trước).

- [ ] **Step 1: Viết test thất bại**

```python
# them vao tests/test_transcribe_backends.py
from transcribe_backends import merge_chunk_entries


def test_merge_dedups_overlap():
    a = [{"start": "00:00:01.000", "end": "00:00:02.000", "text": "A", "words": []},
         {"start": "00:09:59.000", "end": "00:10:00.000", "text": "B", "words": []}]
    b = [{"start": "00:09:59.300", "end": "00:10:00.200", "text": "B", "words": []},  # trung B
         {"start": "00:10:05.000", "end": "00:10:06.000", "text": "C", "words": []}]
    out = merge_chunk_entries([a, b])
    assert [e["text"] for e in out] == ["A", "B", "C"]
```

- [ ] **Step 2: Chạy test — phải FAIL**

Run: `pytest tests/test_transcribe_backends.py -k merge -v`
Expected: FAIL — `ImportError: cannot import name 'merge_chunk_entries'`

- [ ] **Step 3: Thêm hàm**

```python
def _hms_to_sec(hms: str) -> float:
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def merge_chunk_entries(entry_lists: list[list[dict]]) -> list[dict]:
    flat = [e for lst in entry_lists for e in lst]
    flat.sort(key=lambda e: _hms_to_sec(e["start"]))
    merged: list[dict] = []
    for e in flat:
        if merged:
            prev = merged[-1]
            same_text = e["text"].strip() == prev["text"].strip()
            close = abs(_hms_to_sec(e["start"]) - _hms_to_sec(prev["start"])) < 1.0
            if same_text and close:
                continue
        merged.append(e)
    return merged
```

- [ ] **Step 4: Chạy test — phải PASS**

Run: `pytest tests/test_transcribe_backends.py -k merge -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add transcribe_backends.py tests/test_transcribe_backends.py
git commit -m "feat(groq): merge chunk entries with overlap dedup"
```

---

### Task 2.5: `transcribe_groq` orchestration + `load_groq_client`

**Files:**
- Modify: `transcribe_backends.py`
- Test: `tests/test_transcribe_backends.py`

**Interfaces:**
- Consumes: `plan_chunks`, `groq_response_to_entries`, `merge_chunk_entries`
- Produces:
  - `load_groq_client()` — đọc `.env` (nếu có), tạo `groq.Groq()` từ `GROQ_API_KEY`; raise `RuntimeError` nếu thiếu key.
  - `_probe_duration(mp3_path, ffmpeg_exe) -> float` — đọc duration bằng ffmpeg.
  - `transcribe_groq(mp3_path: str, client, ffmpeg_exe: str | None, model: str = "whisper-large-v3-turbo") -> list[dict]` — chia chunk theo duration, với mỗi chunk: ffmpeg cắt+downsample 16k mono FLAC → `client.audio.transcriptions.create(...)` → `groq_response_to_entries(offset)` → `merge_chunk_entries`.

- [ ] **Step 1: Viết test thất bại (mock client + ffmpeg)**

```python
# them vao tests/test_transcribe_backends.py
import transcribe_backends as tb


class FakeTranscriptions:
    def __init__(self, payloads):
        self._payloads = payloads
        self.calls = 0

    def create(self, **kwargs):
        idx = self.calls
        self.calls += 1
        class R:
            def model_dump(self_inner):
                return self._payloads[idx]
        return R()


class FakeAudio:
    def __init__(self, payloads):
        self.transcriptions = FakeTranscriptions(payloads)


class FakeClient:
    def __init__(self, payloads):
        self.audio = FakeAudio(payloads)


def test_transcribe_groq_two_chunks(monkeypatch, tmp_path):
    mp3 = tmp_path / "a.mp3"
    mp3.write_bytes(b"x")

    # duration 1300s -> 3 chunk (600 / 595->1195 / 1190->1300)
    monkeypatch.setattr(tb, "_probe_duration", lambda *a, **k: 1300.0)
    # moi chunk tao file flac gia + tra 1 segment
    monkeypatch.setattr(tb, "_extract_chunk_flac",
                        lambda mp3p, start, end, ffmpeg, outp: outp)
    payloads = [
        {"segments": [{"start": 0.0, "end": 1.0, "text": "chunk0"}]},
        {"segments": [{"start": 0.0, "end": 1.0, "text": "chunk1"}]},
        {"segments": [{"start": 0.0, "end": 1.0, "text": "chunk2"}]},
    ]
    client = FakeClient(payloads)
    out = tb.transcribe_groq(str(mp3), client, "ffmpeg", model="whisper-large-v3")
    texts = [e["text"] for e in out]
    assert texts == ["chunk0", "chunk1", "chunk2"]
    # offset ap dung: chunk1 bat dau ~595s
    assert out[1]["start"].startswith("00:09:5")
```

- [ ] **Step 2: Chạy test — phải FAIL**

Run: `pytest tests/test_transcribe_backends.py -k transcribe_groq -v`
Expected: FAIL — `AttributeError: ... '_probe_duration'`

- [ ] **Step 3: Thêm `load_groq_client`, `_probe_duration`, `_extract_chunk_flac`, `transcribe_groq`**

```python
def load_groq_client():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    import os
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Thieu GROQ_API_KEY")
    from groq import Groq
    return Groq(api_key=key)


def _probe_duration(mp3_path: str, ffmpeg_exe: str) -> float:
    r = subprocess.run(
        [ffmpeg_exe, "-i", mp3_path, "-hide_banner"],
        capture_output=True, text=True, errors="replace",
    )
    import re
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", r.stderr)
    if not m:
        return 0.0
    h, mn, s = m.groups()
    return int(h) * 3600 + int(mn) * 60 + float(s)


def _extract_chunk_flac(mp3_path: str, start: float, end: float,
                        ffmpeg_exe: str, out_path: str) -> str:
    subprocess.run(
        [ffmpeg_exe, "-y", "-loglevel", "error",
         "-ss", str(start), "-to", str(end), "-i", mp3_path,
         "-ar", "16000", "-ac", "1", "-c:a", "flac", out_path],
        check=True, capture_output=True,
    )
    return out_path


def transcribe_groq(mp3_path: str, client, ffmpeg_exe: str | None,
                    model: str = "whisper-large-v3-turbo") -> list[dict]:
    import tempfile, os
    ffmpeg = ffmpeg_exe or "ffmpeg"
    if ffmpeg and Path(ffmpeg).is_dir():
        ffmpeg = str(Path(ffmpeg) / "ffmpeg.exe")
    duration = _probe_duration(mp3_path, ffmpeg)
    windows = plan_chunks(duration) if duration > 0 else [(0.0, 0.0)]

    entry_lists: list[list[dict]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (start, end) in enumerate(windows):
            flac = os.path.join(tmp, f"chunk_{i}.flac")
            try:
                if end > start:
                    _extract_chunk_flac(mp3_path, start, end, ffmpeg, flac)
                    src = flac
                else:
                    src = mp3_path
                with open(src, "rb") as fh:
                    resp = client.audio.transcriptions.create(
                        model=model, file=(os.path.basename(src), fh.read()),
                        language="vi", response_format="verbose_json",
                        timestamp_granularities=["segment", "word"],
                    )
                data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
                entry_lists.append(groq_response_to_entries(data, offset_sec=start))
            except Exception as e:
                logging.error(f"[groq] chunk {i} loi: {e}")
                continue
    return merge_chunk_entries(entry_lists)
```

- [ ] **Step 4: Chạy test — phải PASS**

Run: `pytest tests/test_transcribe_backends.py -k transcribe_groq -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add transcribe_backends.py tests/test_transcribe_backends.py
git commit -m "feat(groq): transcribe_groq orchestration with chunking"
```

---

### Task 2.6: Menu chọn backend Groq trong main.py

**Files:**
- Modify: `main.py` (`main()` — menu backend + load model theo backend)
- Test: thủ công (menu/IO)

**Interfaces:**
- Consumes: `load_groq_client`, `transcribe_groq`, `transcribe_audio`

- [ ] **Step 1: Thêm menu chọn backend (trước menu chọn whisper model)**

Trong `main()`, trước block `whisper_choice = questionary.select(...)`, thêm:

```python
    import os
    has_groq_key = bool(os.environ.get("GROQ_API_KEY"))
    if not has_groq_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            has_groq_key = bool(os.environ.get("GROQ_API_KEY"))
        except Exception:
            pass

    backend_choices = ["WhisperX local (mặc định)"]
    if has_groq_key:
        backend_choices.append("Groq API (nhanh)")
    backend_choice = questionary.select(
        "Backend nhận dạng giọng nói:", choices=backend_choices,
        default="WhisperX local (mặc định)",
    ).ask()
    if backend_choice is None:
        return
    backend = "groq" if backend_choice.startswith("Groq") else "local"
```

- [ ] **Step 2: Phân nhánh chọn model theo backend**

Thay block `whisper_choice = questionary.select(...)` hiện tại bằng:

```python
    if backend == "groq":
        groq_model_choice = questionary.select(
            "Groq Whisper model:",
            choices=["whisper-large-v3-turbo  — nhanh nhất",
                     "whisper-large-v3        — chính xác nhất"],
            default="whisper-large-v3-turbo  — nhanh nhất",
        ).ask()
        if groq_model_choice is None:
            return
        whisper_model = groq_model_choice.split()[0]
    else:
        whisper_choice = questionary.select(
            "Whisper model — tiếng Việt (dùng khi không có caption VTT):",
            choices=[
                "tiny        — nhanh nhất, ít chính xác (~39MB)",
                "base        — cân bằng tốt (~74MB)",
                "small       — chính xác hơn (~244MB)",
                "medium      — rất tốt (~769MB) [mặc định]",
                "large-v2    — tốt, ổn định (~1.5GB)",
                "large-v3    — tốt nhất cho tiếng Việt (~1.5GB)",
            ],
            default="medium      — rất tốt (~769MB) [mặc định]",
        ).ask()
        if whisper_choice is None:
            return
        whisper_model = whisper_choice.split()[0]
```

- [ ] **Step 3: Load model theo backend**

Thay block load model (`wmodel = load_local_model(...)`) bằng:

```python
    console.print(f"\n[bold]🎙 Backend [cyan]{backend}[/cyan] · model [cyan]{whisper_model}[/cyan]…[/bold]")
    if backend == "groq":
        from transcribe_backends import load_groq_client
        wmodel = load_groq_client()
        device = "groq"
    else:
        wmodel = load_local_model(whisper_model)
        device = wmodel["device"]
    console.print(f"[dim]Backend: {backend} [{device}][/dim]\n")
```

- [ ] **Step 4: Truyền `backend` + `whisper_model` vào transcribe worker**

Trong `_transcribe_worker`, sửa lời gọi:

```python
                has_script = transcribe_audio(safe, output_dir, wmodel, backend)
```

Và sửa `transcribe_audio` (Task 2.1) nhánh groq để truyền model name — cập nhật chữ ký gọi:

```python
        if backend == "groq":
            from transcribe_backends import transcribe_groq
            entries = transcribe_groq(str(mp3_path), model, get_ffmpeg_dir(),
                                      model="whisper-large-v3-turbo")
```

> Để truyền đúng tên model Groq người dùng chọn, lưu vào biến module-level: trong `main()` sau khi có `whisper_model`, thêm `globals()["_GROQ_MODEL"] = whisper_model`; trong `transcribe_audio` nhánh groq dùng `model_name = globals().get("_GROQ_MODEL", "whisper-large-v3-turbo")` và truyền `model=model_name`. (Giữ đơn giản, tránh đổi chữ ký `transcribe_audio` đang được worker dùng.)

- [ ] **Step 5: Smoke test import + commit**

Run:
```bash
python -c "import main; print('ok')"
pytest -q
```
Expected: in `ok`; toàn bộ test PASS.

```bash
git add main.py
git commit -m "feat(groq): backend selection menu and model loading in main"
```

---

## Phase 3 — Clean Transcript (#3)

### Task 3.1: Heuristic tagging (music/sound/noise/dialogue)

**Files:**
- Create: `clean_transcript.py`
- Test: `tests/test_clean_transcript.py`

**Interfaces:**
- Produces:
  - Hằng: `TYPE_DIALOGUE="dialogue"`, `TYPE_MUSIC="music"`, `TYPE_SOUND="sound"`, `TYPE_NOISE="noise"`.
  - `is_sound_tag(text: str) -> bool` — match `[Âm nhạc]`, `[Music]`, `[Vỗ tay]`, `[Tiếng cười]`, `♪`, `🎵`, annotation chỉ-trong-ngoặc-vuông.
  - `is_hallucination(text: str) -> bool` — match các câu hallucination thường gặp.
  - `tag_entries_heuristic(entries: list[dict], repeat_threshold: int = 3) -> list[dict]` — trả entries mới có `type`, `text` (=gốc), `text_raw` (=gốc); câu lặp liên tiếp ≥ threshold → `music`.

- [ ] **Step 1: Viết test thất bại**

```python
# tests/test_clean_transcript.py
from clean_transcript import (
    is_sound_tag, is_hallucination, tag_entries_heuristic,
    TYPE_DIALOGUE, TYPE_MUSIC, TYPE_SOUND, TYPE_NOISE,
)


def test_is_sound_tag():
    assert is_sound_tag("[Âm nhạc]")
    assert is_sound_tag("♪ la la la ♪")
    assert is_sound_tag("[Music]")
    assert not is_sound_tag("Xin chào các bạn")


def test_is_hallucination():
    assert is_hallucination("Hãy subscribe cho kênh để xem thêm nhiều video")
    assert is_hallucination("Cảm ơn các bạn đã theo dõi")
    assert not is_hallucination("Hôm nay chúng ta nói về chủ đề này")


def _e(text):
    return {"start": "00:00:00.000", "end": "00:00:01.000", "text": text, "words": []}


def test_tag_entries_types():
    entries = [_e("Xin chào"), _e("[Âm nhạc]"), _e("Hãy subscribe cho kênh nhé")]
    out = tag_entries_heuristic(entries)
    assert out[0]["type"] == TYPE_DIALOGUE
    assert out[1]["type"] == TYPE_SOUND
    assert out[2]["type"] == TYPE_NOISE
    assert out[0]["text_raw"] == "Xin chào"


def test_tag_repetition_as_music():
    entries = [_e("la la la"), _e("la la la"), _e("la la la")]
    out = tag_entries_heuristic(entries, repeat_threshold=3)
    assert all(o["type"] == TYPE_MUSIC for o in out)
```

- [ ] **Step 2: Chạy test — phải FAIL**

Run: `pytest tests/test_clean_transcript.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'clean_transcript'`

- [ ] **Step 3: Tạo `clean_transcript.py` (phần heuristic)**

```python
"""clean_transcript.py — loc nhieu & chuan hoa transcript (heuristic + LLM Groq)."""

import re

TYPE_DIALOGUE = "dialogue"
TYPE_MUSIC = "music"
TYPE_SOUND = "sound"
TYPE_NOISE = "noise"

_SOUND_TAG_RE = re.compile(r"[♪♫\U0001F3B5]|\[[^\]]*\]")
_BRACKET_ONLY_RE = re.compile(r"^\s*\[[^\]]*\]\s*$")

_HALLUCINATION_PATTERNS = [
    "hãy subscribe", "đăng ký kênh", "subscribe cho kênh",
    "cảm ơn các bạn đã theo dõi", "cảm ơn các bạn đã xem",
    "hẹn gặp lại các bạn", "hẹn gặp lại trong video",
    "like và đăng ký", "nhấn chuông thông báo",
]


def is_sound_tag(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if _BRACKET_ONLY_RE.match(t):
        return True
    return bool(re.search(r"[♪♫\U0001F3B5]", t))


def is_hallucination(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in _HALLUCINATION_PATTERNS)


def _norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def tag_entries_heuristic(entries: list[dict], repeat_threshold: int = 3) -> list[dict]:
    out: list[dict] = []
    for e in entries:
        raw = e.get("text", "")
        new = dict(e)
        new["text_raw"] = raw
        new["text"] = raw
        if is_sound_tag(raw):
            new["type"] = TYPE_SOUND
        elif is_hallucination(raw):
            new["type"] = TYPE_NOISE
        else:
            new["type"] = TYPE_DIALOGUE
        out.append(new)

    # cau lap lien tiep >= threshold -> music
    i = 0
    n = len(out)
    while i < n:
        j = i
        key = _norm_key(out[i]["text_raw"])
        while j + 1 < n and _norm_key(out[j + 1]["text_raw"]) == key and key:
            j += 1
        if (j - i + 1) >= repeat_threshold:
            for k in range(i, j + 1):
                if out[k]["type"] == TYPE_DIALOGUE:
                    out[k]["type"] = TYPE_MUSIC
        i = j + 1
    return out
```

- [ ] **Step 4: Chạy test — phải PASS**

Run: `pytest tests/test_clean_transcript.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add clean_transcript.py tests/test_clean_transcript.py
git commit -m "feat(clean): heuristic tagging of transcript segments"
```

---

### Task 3.2: LLM batch classify + normalize (mock Groq)

**Files:**
- Modify: `clean_transcript.py`
- Test: `tests/test_clean_transcript.py`

**Interfaces:**
- Consumes: hằng TYPE_*
- Produces:
  - `build_llm_prompt(batch: list[dict]) -> str` — tạo prompt liệt kê index + text.
  - `apply_llm_result(batch: list[dict], result: list[dict]) -> list[dict]` — map theo `index`; cập nhật `type` và `text` (chuẩn hóa); item ngoài phạm vi/sai index thì giữ nguyên.
  - `llm_clean_batch(client, batch: list[dict], model="llama-3.1-8b-instant") -> list[dict]` — gọi chat completions JSON mode, parse, gọi `apply_llm_result`; lỗi → trả `batch` nguyên.

- [ ] **Step 1: Viết test thất bại**

```python
# them vao tests/test_clean_transcript.py
import json
from clean_transcript import build_llm_prompt, apply_llm_result, llm_clean_batch


def _te(text, typ="dialogue"):
    return {"start": "00:00:00.000", "end": "00:00:01.000",
            "text": text, "text_raw": text, "type": typ, "words": []}


def test_build_prompt_lists_indexes():
    p = build_llm_prompt([_te("Xin chao"), _te("la la la")])
    assert "0" in p and "Xin chao" in p and "la la la" in p


def test_apply_llm_result_updates():
    batch = [_te("xin chao cac ban"), _te("la la la")]
    result = [{"index": 0, "type": "dialogue", "text": "Xin chào các bạn."},
              {"index": 1, "type": "music", "text": "la la la"}]
    out = apply_llm_result(batch, result)
    assert out[0]["text"] == "Xin chào các bạn."
    assert out[0]["type"] == "dialogue"
    assert out[1]["type"] == "music"
    assert out[0]["text_raw"] == "xin chao cac ban"     # giu goc


def test_llm_clean_batch_error_returns_input(monkeypatch):
    class BadClient:
        class chat:
            class completions:
                @staticmethod
                def create(**k):
                    raise RuntimeError("boom")
    batch = [_te("a")]
    out = llm_clean_batch(BadClient(), batch)
    assert out == batch
```

- [ ] **Step 2: Chạy test — phải FAIL**

Run: `pytest tests/test_clean_transcript.py -k "prompt or apply_llm or llm_clean" -v`
Expected: FAIL — `ImportError: cannot import name 'build_llm_prompt'`

- [ ] **Step 3: Thêm 3 hàm**

```python
import json


def build_llm_prompt(batch: list[dict]) -> str:
    lines = [f"{i}: {e.get('text_raw', e.get('text',''))}" for i, e in enumerate(batch)]
    return (
        "Bạn là bộ lọc transcript tiếng Việt. Với mỗi dòng dưới đây, "
        "phân loại 'type' là một trong: dialogue (lời thoại/nội dung nói), "
        "music (lời bài hát/giai điệu), noise (câu thừa, hallucination, "
        "lời chào câu view không thuộc nội dung). Đồng thời chuẩn hóa 'text': "
        "sửa dấu câu, viết hoa đầu câu, bỏ từ đệm lặp. "
        "Trả về JSON: {\"items\": [{\"index\": int, \"type\": str, \"text\": str}, ...]}.\n\n"
        + "\n".join(lines)
    )


def apply_llm_result(batch: list[dict], result: list[dict]) -> list[dict]:
    out = [dict(e) for e in batch]
    for item in result:
        try:
            idx = int(item["index"])
        except (KeyError, ValueError, TypeError):
            continue
        if 0 <= idx < len(out):
            if item.get("type") in (TYPE_DIALOGUE, TYPE_MUSIC, TYPE_SOUND, TYPE_NOISE):
                out[idx]["type"] = item["type"]
            if isinstance(item.get("text"), str) and item["text"].strip():
                out[idx]["text"] = item["text"].strip()
    return out


def llm_clean_batch(client, batch: list[dict],
                    model: str = "llama-3.1-8b-instant") -> list[dict]:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": build_llm_prompt(batch)}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        items = data.get("items", data if isinstance(data, list) else [])
        return apply_llm_result(batch, items)
    except Exception:
        return batch
```

- [ ] **Step 4: Chạy test — phải PASS**

Run: `pytest tests/test_clean_transcript.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add clean_transcript.py tests/test_clean_transcript.py
git commit -m "feat(clean): LLM batch classify and normalize via Groq"
```

---

### Task 3.3: `clean_entries` + `clean_file` (ghi .clean.json)

**Files:**
- Modify: `clean_transcript.py`
- Test: `tests/test_clean_transcript.py`

**Interfaces:**
- Consumes: `tag_entries_heuristic`, `llm_clean_batch`
- Produces:
  - `clean_entries(entries: list[dict], client=None, batch_size=40) -> list[dict]` — heuristic trước; nếu `client` có thì chạy LLM theo lô.
  - `clean_file(json_path, client=None) -> "Path"` — đọc `<name>.json`, `clean_entries`, ghi `<name>.clean.json`; trả path output. KHÔNG đụng file gốc.

- [ ] **Step 1: Viết test thất bại**

```python
# them vao tests/test_clean_transcript.py
import json
from pathlib import Path
from clean_transcript import clean_entries, clean_file


def test_clean_entries_heuristic_only():
    entries = [{"start": "00:00:00.000", "end": "00:00:01.000", "text": "[Âm nhạc]", "words": []},
               {"start": "00:00:01.000", "end": "00:00:02.000", "text": "Xin chào", "words": []}]
    out = clean_entries(entries, client=None)
    assert out[0]["type"] == "sound"
    assert out[1]["type"] == "dialogue"


def test_clean_file_writes_clean_json(tmp_path):
    src = tmp_path / "a.json"
    src.write_text(json.dumps(
        [{"start": "00:00:00.000", "end": "00:00:01.000", "text": "Xin chào", "words": []}],
        ensure_ascii=False), encoding="utf-8")
    out = clean_file(src, client=None)
    assert out == tmp_path / "a.clean.json"
    assert out.exists()
    assert src.exists()      # goc khong bi dung
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data[0]["type"] == "dialogue"
```

- [ ] **Step 2: Chạy test — phải FAIL**

Run: `pytest tests/test_clean_transcript.py -k "clean_entries or clean_file" -v`
Expected: FAIL — `ImportError: cannot import name 'clean_entries'`

- [ ] **Step 3: Thêm 2 hàm**

```python
from pathlib import Path


def clean_entries(entries: list[dict], client=None, batch_size: int = 40) -> list[dict]:
    tagged = tag_entries_heuristic(entries)
    if client is None:
        return tagged
    out: list[dict] = []
    for i in range(0, len(tagged), batch_size):
        out.extend(llm_clean_batch(client, tagged[i:i + batch_size]))
    return out


def clean_file(json_path, client=None) -> Path:
    json_path = Path(json_path)
    entries = json.loads(json_path.read_text(encoding="utf-8"))
    cleaned = clean_entries(entries, client=client)
    out_path = json_path.with_suffix(".clean.json")
    out_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
```

- [ ] **Step 4: Chạy test — phải PASS**

Run: `pytest tests/test_clean_transcript.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add clean_transcript.py tests/test_clean_transcript.py
git commit -m "feat(clean): clean_entries and clean_file writing .clean.json"
```

---

### Task 3.4: CLI standalone cho clean_transcript.py

**Files:**
- Modify: `clean_transcript.py` (thêm `main()` + `__main__`)
- Test: thủ công

**Interfaces:**
- Consumes: `clean_file`, `load_groq_client` (từ transcribe_backends)

- [ ] **Step 1: Thêm `main()` + entrypoint**

```python
def _collect_json(paths):
    result = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            result.extend(f for f in sorted(p.glob("*.json"))
                          if not f.name.endswith(".clean.json") and f.name != "manifest.json")
        elif p.suffix.lower() == ".json" and p.exists() and not p.name.endswith(".clean.json"):
            result.append(p)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Loc nhieu & chuan hoa transcript JSON.")
    parser.add_argument("inputs", nargs="*", default=["downloads"])
    parser.add_argument("--no-llm", action="store_true", help="Chi heuristic, khong goi Groq")
    args = parser.parse_args()

    client = None
    if not args.no_llm:
        try:
            from transcribe_backends import load_groq_client
            client = load_groq_client()
            print("[clean] Dung LLM Groq de chuan hoa.")
        except Exception as e:
            print(f"[clean] Khong co Groq ({e}) -> chi heuristic.")

    files = _collect_json(args.inputs or ["downloads"])
    if not files:
        print("Khong tim thay file JSON.")
        return
    for jf in files:
        out = clean_file(jf, client=client)
        print(f"  ✓ {jf.name} -> {out.name}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test trên dữ liệu thật (heuristic)**

Run:
```bash
python clean_transcript.py --no-llm "downloads/Thế nào là một cuộc đời đáng sống_ _ Sách Nhân Gian Đáng Giá.json"
```
Expected: in `✓ ... -> ....clean.json`; file `.clean.json` xuất hiện cạnh file gốc, file gốc còn nguyên.

- [ ] **Step 3: Commit**

```bash
git add clean_transcript.py
git commit -m "feat(clean): standalone CLI for clean_transcript"
```

---

### Task 3.5: Gọi clean inline trong main.py

**Files:**
- Modify: `main.py` (`_transcribe_worker` — gọi clean sau khi có .json)

**Interfaces:**
- Consumes: `clean_transcript.clean_file`, Groq client (nếu backend groq / có key)

- [ ] **Step 1: Tạo clean client 1 lần trong `main()` (sau khi xác định backend)**

Sau block load model trong `main()`, thêm:

```python
    clean_client = None
    try:
        if backend == "groq":
            clean_client = wmodel        # tai dung client groq da co
        elif has_groq_key:
            from transcribe_backends import load_groq_client
            clean_client = load_groq_client()
    except Exception:
        clean_client = None
```

- [ ] **Step 2: Gọi clean trong `_transcribe_worker` sau khi transcribe thành công**

Trong `_transcribe_worker`, sau dòng `has_script = transcribe_audio(...)`, thêm:

```python
                if has_script:
                    try:
                        from clean_transcript import clean_file
                        clean_file(Path(output_dir) / f"{safe}.json", client=clean_client)
                    except Exception:
                        pass
```

- [ ] **Step 3: Smoke test import + commit**

Run:
```bash
python -c "import main; print('ok')"
pytest -q
```
Expected: in `ok`; test PASS.

```bash
git add main.py
git commit -m "feat(clean): run clean step inline after transcription"
```

---

### Task 3.6: split_audio.py ưu tiên .clean.json + chỉ cắt dialogue

**Files:**
- Modify: `split_audio.py` (`split_one` — chọn nguồn entries)
- Test: `tests/test_split_audio_clean.py`

**Interfaces:**
- Produces:
  - `load_entries_for_split(json_path: "Path") -> list[dict]` — nếu có `<stem>.clean.json` thì đọc nó và **chỉ giữ** `type=="dialogue"`; ngược lại đọc `json_path` nguyên.

- [ ] **Step 1: Viết test thất bại**

```python
# tests/test_split_audio_clean.py
import json
from pathlib import Path
import split_audio


def test_prefers_clean_and_filters_dialogue(tmp_path):
    base = tmp_path / "a.json"
    base.write_text(json.dumps([
        {"start": "00:00:00.000", "end": "00:00:01.000", "text": "raw1", "words": []},
        {"start": "00:00:01.000", "end": "00:00:02.000", "text": "raw2", "words": []},
    ], ensure_ascii=False), encoding="utf-8")
    clean = tmp_path / "a.clean.json"
    clean.write_text(json.dumps([
        {"start": "00:00:00.000", "end": "00:00:01.000", "text": "Thoai", "type": "dialogue", "words": []},
        {"start": "00:00:01.000", "end": "00:00:02.000", "text": "[Âm nhạc]", "type": "sound", "words": []},
    ], ensure_ascii=False), encoding="utf-8")

    entries = split_audio.load_entries_for_split(base)
    assert len(entries) == 1
    assert entries[0]["text"] == "Thoai"


def test_falls_back_to_json(tmp_path):
    base = tmp_path / "b.json"
    base.write_text(json.dumps([
        {"start": "00:00:00.000", "end": "00:00:01.000", "text": "x", "words": []},
    ], ensure_ascii=False), encoding="utf-8")
    entries = split_audio.load_entries_for_split(base)
    assert len(entries) == 1
    assert entries[0]["text"] == "x"
```

- [ ] **Step 2: Chạy test — phải FAIL**

Run: `pytest tests/test_split_audio_clean.py -v`
Expected: FAIL — `AttributeError: module 'split_audio' has no attribute 'load_entries_for_split'`

- [ ] **Step 3: Thêm `load_entries_for_split` + dùng trong `split_one`**

Thêm hàm vào `split_audio.py`:

```python
def load_entries_for_split(json_path: Path) -> list[dict]:
    clean = json_path.with_suffix(".clean.json")
    if clean.exists():
        data = json.loads(clean.read_text(encoding="utf-8"))
        return [e for e in data if e.get("type", "dialogue") == "dialogue"]
    return json.loads(json_path.read_text(encoding="utf-8"))
```

Trong `split_one`, thay dòng:

```python
    entries: list[dict] = json.loads(json_path.read_text(encoding="utf-8"))
```

bằng:

```python
    entries: list[dict] = load_entries_for_split(json_path)
```

- [ ] **Step 4: Chạy test — phải PASS**

Run: `pytest tests/test_split_audio_clean.py -v`
Expected: PASS

- [ ] **Step 5: Smoke test toàn bộ + commit**

Run:
```bash
python -c "import split_audio; print('ok')"
pytest -q
```
Expected: in `ok`; toàn bộ test PASS.

```bash
git add split_audio.py tests/test_split_audio_clean.py
git commit -m "feat(split): prefer .clean.json and cut only dialogue segments"
```

---

## Cập nhật tài liệu (cuối cùng)

### Task 4: Cập nhật README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Thêm mục mô tả 3 tính năng mới**

Thêm vào `README.md` (sau mục "Tính năng") các gạch đầu dòng:
- Proxy pool xoay IP free/custom (bật trong menu) để crawl số lượng lớn.
- Lựa chọn backend Groq API cho Whisper (cần `GROQ_API_KEY`).
- Bước lọc nhiễu/chuẩn hóa transcript (`clean_transcript.py`), sinh `.clean.json`.

Và mục cấu hình:
- Đặt `GROQ_API_KEY` trong biến môi trường hoặc file `.env`.
- (Tùy chọn) tạo `proxies.txt` để nạp proxy riêng.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document proxy pool, Groq backend, and clean step"
```

---

## Self-Review (đã thực hiện khi viết plan)

**Spec coverage:**
- #1 proxy auto free + custom + background refresh + xoay khi 429 → Task 1.1–1.6. ✓
- #2 Groq backend tùy chọn, local mặc định, chunk file dài, gate bằng key → Task 2.1–2.6. ✓
- #3 hybrid heuristic + LLM, tag giữ bản gốc (.clean.json), inline + standalone, split ưu tiên clean → Task 3.1–3.6. ✓
- Dependencies + test infra → Task 0. ✓
- Tương thích ngược (proxy off, local default, fallback json) → ràng buộc trong Task 1.6/2.6/3.6. ✓

**Type consistency:** `ProxyPool` API (`get_proxy`/`mark_bad`/`refresh`/`set_alive`/`start_background`/`stop_background`) nhất quán giữa Task 1.2–1.6. Schema entries (`start/end/text/words`, thêm `type/text_raw`) nhất quán giữa Phase 2 & 3 & split. `transcribe_audio(safe, output_dir, model, backend)` khớp giữa Task 2.1, 2.6, 3.5.

**Placeholder scan:** không có TBD/TODO; mọi step có code/command cụ thể.
