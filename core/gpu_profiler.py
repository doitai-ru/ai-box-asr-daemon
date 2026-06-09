# -*- coding: utf-8 -*-
"""
GPU-профилировщик: лёгкая выключаемая инструментация.

Гейт settings.GPU_PROFILE. При выкл — record/profile это no-op, накладных ноль.
«Текущую память» компонентные записи берут из кэш-снимка _snapshot, который
обновляет фон-таска gpu_sampler_loop (NVML не на горячем пути). Атрибуция — по
корреляции во времени (пишем компонент+действие+текущая память), без дельт.
"""

import json
import logging
import logging.handlers
import os
import time
from contextlib import asynccontextmanager, contextmanager

from config import settings

logger = logging.getLogger(__name__)

# Кэш-снимок (обновляется gpu_sampler_loop)
_snapshot = {
    "process_gpu_mib": None, "gpu_used_mib": None, "gpu_free_mib": None,
    "gpu_total_mib": None, "gpu_util_pct": None, "updated_at": None,
}
_components = {}   # компонент -> {active, calls, last_action, last_ms}
_queues = set()    # активные очереди (бэклог)
_file_logger = None
_nvml = {"inited": False, "handle": None}


def enabled() -> bool:
    return bool(getattr(settings, "GPU_PROFILE", False))


def reset() -> None:
    """Сброс состояния (для тестов)."""
    _components.clear()
    _queues.clear()


def _get_file_logger():
    global _file_logger
    if _file_logger is None:
        lg = logging.getLogger("gpu_profile")
        lg.setLevel(logging.INFO)
        lg.propagate = False
        if not lg.handlers:
            try:
                os.makedirs("logs", exist_ok=True)
                h = logging.handlers.RotatingFileHandler(
                    "logs/gpu_profile.jsonl", maxBytes=50 * 1024 * 1024,
                    backupCount=3, encoding="utf-8")
                h.setFormatter(logging.Formatter("%(message)s"))
                lg.addHandler(h)
            except Exception:
                logger.warning("gpu_profile file logger init failed", exc_info=True)
        _file_logger = lg
    return _file_logger


def _write(obj: dict) -> None:
    try:
        _get_file_logger().info(json.dumps(obj, ensure_ascii=False))
    except Exception:
        pass


def record(component: str, action: str, **fields) -> None:
    if not enabled():
        return
    comp = _components.setdefault(
        component, {"active": 0, "calls": 0, "last_action": None, "last_ms": None})
    if action == "start":
        comp["active"] += 1
    elif action == "end":
        comp["active"] = max(0, comp["active"] - 1)
        comp["calls"] += 1
        if "dur_ms" in fields:
            comp["last_ms"] = fields["dur_ms"]
    comp["last_action"] = action
    _write({"ts": time.time(), "component": component, "action": action,
            "gpu_mib": _snapshot.get("process_gpu_mib"), **fields})


@contextmanager
def profile_block(component: str, **fields):
    """Sync-контекст вокруг инференса (VAD/диар/GigaAM выполняются синхронно)."""
    if not enabled():
        yield
        return
    record(component, "start", **fields)
    t0 = time.time()
    try:
        yield
    finally:
        record(component, "end", dur_ms=round((time.time() - t0) * 1000, 1))


@asynccontextmanager
async def profile(component: str, **fields):
    """Async-контекст вокруг await-инференса (пунктуация/T-one)."""
    if not enabled():
        yield
        return
    record(component, "start", **fields)
    t0 = time.time()
    try:
        yield
    finally:
        record(component, "end", dur_ms=round((time.time() - t0) * 1000, 1))


def snapshot() -> dict:
    return dict(_snapshot)


def components() -> dict:
    return {k: dict(v) for k, v in _components.items()}


def register_queue(q) -> None:
    _queues.add(q)


def unregister_queue(q) -> None:
    _queues.discard(q)


def tone_backlog() -> int:
    total = 0
    for q in list(_queues):
        try:
            total += q.qsize()
        except Exception:
            pass
    return total


def read_gpu_snapshot() -> dict:
    """NVML: память своего процесса (PID) + used/free/total/util карты. Без GPU -> None-поля."""
    out = {"process_gpu_mib": None, "gpu_used_mib": None, "gpu_free_mib": None,
           "gpu_total_mib": None, "gpu_util_pct": None, "updated_at": time.time()}
    try:
        import pynvml
        if not _nvml["inited"]:
            pynvml.nvmlInit()
            _nvml["handle"] = pynvml.nvmlDeviceGetHandleByIndex(0)
            _nvml["inited"] = True
        h = _nvml["handle"]
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        out["gpu_used_mib"] = int(mem.used // (1024 * 1024))
        out["gpu_free_mib"] = int(mem.free // (1024 * 1024))
        out["gpu_total_mib"] = int(mem.total // (1024 * 1024))
        try:
            out["gpu_util_pct"] = int(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
        except Exception:
            pass
        try:
            mypid = os.getpid()
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
                if p.pid == mypid:
                    out["process_gpu_mib"] = int((p.usedGpuMemory or 0) // (1024 * 1024))
                    break
        except Exception:
            pass
    except Exception:
        logger.debug("NVML недоступен", exc_info=True)
    return out


async def gpu_sampler_loop(app_state, interval_sec: float = 1.0) -> None:
    """Фон-таска: раз в interval_sec обновляет _snapshot и пишет sample-строку."""
    import asyncio
    while True:
        try:
            snap = read_gpu_snapshot()
            _snapshot.update(snap)
            conns = {}
            mgr = getattr(app_state, "ws_manager", None)
            if mgr is not None and hasattr(mgr, "counts_by_kind"):
                conns = mgr.counts_by_kind()
            _write({"ts": time.time(), "kind": "sample",
                    "conns_by_path": conns, "tone_backlog": tone_backlog(), **snap})
        except Exception:
            logger.warning("gpu_sampler_loop ошибка", exc_info=True)
        await asyncio.sleep(interval_sec)
