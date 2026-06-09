# -*- coding: utf-8 -*-
"""Тесты GPU-профилировщика (ядро, реестр очередей, счётчики)."""

import asyncio
import queue

import pytest

from core import gpu_profiler as gp


@pytest.fixture(autouse=True)
def _reset():
    gp.reset()
    yield
    gp.reset()


def test_disabled_is_noop(monkeypatch):
    monkeypatch.setattr("config.settings.GPU_PROFILE", False)
    with gp.profile_block("vad", n=512):
        pass
    assert gp.components() == {}            # выкл -> ничего не пишем


def test_enabled_profile_block_moves_counters(monkeypatch):
    monkeypatch.setattr("config.settings.GPU_PROFILE", True)
    with gp.profile_block("giga", samples=480000):
        assert gp.components()["giga"]["active"] == 1   # внутри — активен
    c = gp.components()["giga"]
    assert c["active"] == 0 and c["calls"] == 1          # после — счётчик
    assert isinstance(c["last_ms"], float)


def test_enabled_async_profile(monkeypatch):
    monkeypatch.setattr("config.settings.GPU_PROFILE", True)

    async def go():
        async with gp.profile("tone", n=2400):
            assert gp.components()["tone"]["active"] == 1
    asyncio.run(go())
    assert gp.components()["tone"]["calls"] == 1


def test_queue_backlog(monkeypatch):
    monkeypatch.setattr("config.settings.GPU_PROFILE", True)
    q1, q2 = queue.Queue(), queue.Queue()
    q1.put(1); q1.put(2); q2.put(3)
    gp.register_queue(q1); gp.register_queue(q2)
    assert gp.tone_backlog() == 3
    gp.unregister_queue(q1)
    assert gp.tone_backlog() == 1


def test_read_gpu_snapshot_no_nvml(monkeypatch):
    # эмулируем отсутствие pynvml: read_gpu_snapshot не падает, поля None
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "pynvml":
            raise ImportError("no pynvml")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    snap = gp.read_gpu_snapshot()
    assert snap["process_gpu_mib"] is None
    assert "updated_at" in snap and snap["updated_at"] > 0


def test_connection_manager_counts_by_kind():
    from services.ws_manager import ConnectionManager

    class FakeWS:
        async def accept(self): pass
        async def close(self, code=1000, reason=""): pass

    async def go():
        mgr = ConnectionManager(max_connections=10)
        await mgr.connect(FakeWS(), "a", kind="giga")
        await mgr.connect(FakeWS(), "b", kind="tone")
        await mgr.connect(FakeWS(), "c", kind="tone")
        return mgr.counts_by_kind()

    counts = asyncio.run(go())
    assert counts["giga"] == 1
    assert counts["tone"] == 2
    assert counts["total"] == 3


def test_gpu_profile_endpoint_disabled(monkeypatch):
    monkeypatch.setattr("config.settings.GPU_PROFILE", False)
    from api.v1.endpoints.admin import admin_gpu_profile

    class FakeReq:
        class app:
            class state:
                ws_manager = None

    res = asyncio.run(admin_gpu_profile(current_user=object(), request=FakeReq))
    assert res == {"enabled": False}


def test_gpu_profile_endpoint_enabled(monkeypatch):
    monkeypatch.setattr("config.settings.GPU_PROFILE", True)
    from api.v1.endpoints.admin import admin_gpu_profile

    class FakeMgr:
        def counts_by_kind(self): return {"tone": 2, "total": 2}

    class FakeReq:
        class app:
            class state:
                ws_manager = FakeMgr()

    res = asyncio.run(admin_gpu_profile(current_user=object(), request=FakeReq))
    assert res["enabled"] is True
    assert "snapshot" in res and "components" in res
    assert res["conns_by_path"] == {"tone": 2, "total": 2}
    assert "config" in res and "tone_infer_workers" in res["config"]


def test_monitor_analyze_plateau_vs_growth():
    from tools.gpu_monitor import analyze
    plateau = analyze([100, 200, 300, 300, 300, 300])
    assert plateau["max"] == 300
    assert plateau["verdict"] == "plateau"
    growth = analyze([100, 200, 300, 400, 500, 600])
    assert growth["verdict"] == "growing"
