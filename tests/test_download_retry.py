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
