# GPU-профилировщик — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Лёгкая выключаемая инструментация GPU: временной ряд «память процесса + конкуренция по путям + бэклог + активный компонент» для мониторинга боя, плюс эндпоинт-снимок и standalone-монитор.

**Architecture:** Модуль `core/gpu_profiler.py` держит кэш-снимок (обновляет фон-сэмплер раз в ~1с, NVML не на горячем пути), счётчики по компонентам и реестр очередей. Компоненты оборачивают инференс в `profile()`/`profile_block()` (no-op при `GPU_PROFILE=False`). Эндпоинт `/api/v1/admin/gpu-profile` отдаёт снимок; `tools/gpu_monitor.py` снимает кривую.

**Tech Stack:** Python 3.12, pynvml~=12 (есть в requirements), FastAPI, asyncio, pytest 9 (тесты синхронные через `asyncio.run`, без pytest-asyncio в части кейсов).

**Спек:** `docs/superpowers/specs/2026-06-09-gpu-profiler-design.md`

**Команды:** интерпретатор `venv/bin/python`, тесты `venv/bin/pytest`. Запуск нового теста: `venv/bin/pytest tests/test_gpu_profiler.py -v`.

---

## Структура файлов

- **Создаём** `core/gpu_profiler.py` — ядро: гейт, кэш-снимок, `record`/`profile`/`profile_block`, агрегаты по компонентам, реестр очередей (`register_queue`/`tone_backlog`), NVML-чтение (`read_gpu_snapshot`) и фон-таска (`gpu_sampler_loop`).
- **Изменяем** `config.py` — `GPU_PROFILE: bool = False`.
- **Изменяем** `services/ws_manager.py` — `ConnectionMeta.kind`, параметр `kind` у `connect()`, метод `counts_by_kind()`.
- **Изменяем** `main.py` — старт/останов `gpu_sampler_loop` в lifespan при `GPU_PROFILE`.
- **Изменяем** компоненты (обёртка инференса + регистрация очереди): `VoiceActivityDetector/do_vad.py`, `Diarisation/do_diarize.py`, `Punctuation/__init__.py`, `Recognizer/engine/stream_recognition.py`, `utils/tone_stream.py`, `api/v1/endpoints/asr_ws_tone.py`, `api/v1/endpoints/asr_ws.py`.
- **Изменяем** `api/v1/endpoints/admin.py` — эндпоинт `GET /gpu-profile`.
- **Создаём** `tools/gpu_monitor.py` — standalone-монитор.
- **Создаём** `tests/test_gpu_profiler.py` — юнит-тесты ядра/реестра/счётчиков/NVML(mock)/анализа.

---

## Task 1: Ядро профилировщика + config-гейт

**Files:**
- Modify: `config.py` (после `TONE_INFER_WORKERS`)
- Create: `core/gpu_profiler.py`
- Create: `tests/test_gpu_profiler.py`

- [ ] **Step 1: Падающий тест**

Создать `tests/test_gpu_profiler.py`:

```python
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
```

- [ ] **Step 2: Запустить — упадёт**

Run: `venv/bin/pytest tests/test_gpu_profiler.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.gpu_profiler'`.

- [ ] **Step 3: Добавить настройку в `config.py`**

После строки `TONE_INFER_WORKERS: int = 1` (в блоке T-one settings) вставить:

```python
    # GPU-профилировщик: при True включается фон-сэмплер NVML + инструментация
    # компонентов (профиль пишется в logs/gpu_profile.jsonl). По умолчанию выкл —
    # нулевые накладные. Включать только на расследование памяти.
    GPU_PROFILE: bool = False
```

И добавить `'GPU_PROFILE'` в список полей валидатора `_int_to_bool` (там же, где `STREAM_WITH_GPU`):

```python
        'HUMAN_FORMAT_MD_FILE', 'USE_TONE_STREAMING', 'STREAM_WITH_GPU', 'GPU_PROFILE',
```

- [ ] **Step 4: Создать `core/gpu_profiler.py`**

```python
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
```

- [ ] **Step 5: Запустить — пройдёт**

Run: `venv/bin/pytest tests/test_gpu_profiler.py -q`
Expected: PASS (4 passed).

- [ ] **Step 6: Коммит**

```bash
git add config.py core/gpu_profiler.py tests/test_gpu_profiler.py
git commit -m "$(cat <<'EOF'
GPU-профиль: ядро (gpu_profiler) + гейт GPU_PROFILE

profile()/profile_block()/record() (no-op при выкл), агрегаты по компонентам,
реестр очередей (tone_backlog), NVML-снимок и фон-сэмплер. Профиль в JSONL.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: NVML-снимок без GPU (mock) + устойчивость

**Files:**
- Modify: `tests/test_gpu_profiler.py` (дописать)

- [ ] **Step 1: Падающий тест (поведение без GPU/NVML)**

Дописать в `tests/test_gpu_profiler.py`:

```python
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
```

- [ ] **Step 2: Запустить — должен пройти сразу (код уже устойчив)**

Run: `venv/bin/pytest tests/test_gpu_profiler.py::test_read_gpu_snapshot_no_nvml -q`
Expected: PASS — `read_gpu_snapshot` ловит ImportError и возвращает None-поля.

(Этот тест фиксирует контракт устойчивости из Task 1; реализация уже есть. Если падает — значит исключение не ловится, чинить в `read_gpu_snapshot`.)

- [ ] **Step 3: Коммит**

```bash
git add tests/test_gpu_profiler.py
git commit -m "$(cat <<'EOF'
GPU-профиль: тест устойчивости read_gpu_snapshot без NVML

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: ConnectionManager — kind + counts_by_kind

**Files:**
- Modify: `services/ws_manager.py` (ConnectionMeta lines 20-41, connect lines 76-97)
- Modify: `tests/test_gpu_profiler.py` (дописать тест)

- [ ] **Step 1: Падающий тест**

Дописать в `tests/test_gpu_profiler.py`:

```python
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
```

- [ ] **Step 2: Запустить — упадёт**

Run: `venv/bin/pytest tests/test_gpu_profiler.py::test_connection_manager_counts_by_kind -q`
Expected: FAIL — `connect()` не принимает `kind` / нет `counts_by_kind`.

- [ ] **Step 3: Добавить поле `kind` в `ConnectionMeta`**

В `services/ws_manager.py`, в dataclass `ConnectionMeta` (после `user_id: Optional[str] = None`):

```python
    kind: str = "generic"
```

- [ ] **Step 4: Параметр `kind` в `connect()` + метод `counts_by_kind()`**

В `services/ws_manager.py` заменить сигнатуру и тело `connect`:

```python
    async def connect(self, websocket: WebSocket, client_id: str, kind: str = "generic") -> bool:
```

и строку создания меты:

```python
        self.connection_meta[client_id] = ConnectionMeta(kind=kind)
```

Сразу после свойства `active_connections_count` (строки 71-74) добавить:

```python
    def counts_by_kind(self) -> dict:
        """Число активных соединений по путям (giga/tone/admin/generic) + total."""
        out: dict = {}
        for meta in self.connection_meta.values():
            k = getattr(meta, "kind", "generic")
            out[k] = out.get(k, 0) + 1
        out["total"] = self.active_connections_count
        return out
```

- [ ] **Step 5: Запустить — пройдёт**

Run: `venv/bin/pytest tests/test_gpu_profiler.py -q`
Expected: PASS (все тесты файла).

- [ ] **Step 6: Коммит**

```bash
git add services/ws_manager.py tests/test_gpu_profiler.py
git commit -m "$(cat <<'EOF'
ws_manager: ConnectionMeta.kind + connect(kind=) + counts_by_kind()

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Старт/останов сэмплера в lifespan

**Files:**
- Modify: `main.py` (импорт; старт после metrics_task ~line 119; останов в shutdown ~line 186)

Примечание: lifespan грузит модели — юнит-тестом не покрываем, проверяем компиляцией + импортом.

- [ ] **Step 1: Импорт**

В `main.py` рядом с прочими `from core...` импортами добавить:

```python
from core import gpu_profiler
```

- [ ] **Step 2: Старт сэмплера**

В `lifespan`, сразу ПОСЛЕ блока создания `app.state.metrics_task = asyncio.create_task(metrics_reporter_loop(...))` (строки 116-118), добавить:

```python
    # GPU-профилировщик (фон-сэмплер) — только при GPU_PROFILE
    if settings.GPU_PROFILE:
        app.state.gpu_profile_task = asyncio.create_task(
            gpu_profiler.gpu_sampler_loop(app.state, interval_sec=1.0)
        )
        logger.info("GPU-профилировщик включён (сэмплер 1с -> logs/gpu_profile.jsonl)")
```

- [ ] **Step 3: Останов сэмплера**

В секции shutdown, сразу ПОСЛЕ остановки `metrics_task` (строки 181-186), добавить:

```python
    if hasattr(app.state, "gpu_profile_task"):
        app.state.gpu_profile_task.cancel()
        try:
            await app.state.gpu_profile_task
        except asyncio.CancelledError:
            pass
```

- [ ] **Step 4: Проверка**

Run: `venv/bin/python -m py_compile main.py core/gpu_profiler.py && venv/bin/python -c "import main; print('import main OK')"`
Expected: вывод `import main OK` (предупреждения WARNING:root — норма).

- [ ] **Step 5: Коммит**

```bash
git add main.py
git commit -m "$(cat <<'EOF'
GPU-профиль: старт/останов gpu_sampler_loop в lifespan (при GPU_PROFILE)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Инструментация компонентов + регистрация очереди

**Files:**
- Modify: `VoiceActivityDetector/do_vad.py` (line 87), `Diarisation/do_diarize.py` (line 207), `Punctuation/__init__.py` (line 142-150), `Recognizer/engine/stream_recognition.py`, `utils/tone_stream.py`, `api/v1/endpoints/asr_ws_tone.py` (call site + queue), `api/v1/endpoints/asr_ws.py` (call site)

Все обёртки — no-op при `GPU_PROFILE=False`, инференс не затрагивается. Покрытие — компиляцией + живым прогоном (Task 8); существующий `tests/test_tone_offload.py` подтверждает, что обёртка T-one не ломает поведение.

- [ ] **Step 1: VAD** — `VoiceActivityDetector/do_vad.py`

Добавить импорт вверху файла:
```python
from core import gpu_profiler
```
Обернуть инференс (строка 87 `outputs = self.session.run(['output', 'stateN'], inputs)`):
```python
        with gpu_profiler.profile_block("vad", n=int(len(audio_frame))):
            outputs = self.session.run(['output', 'stateN'], inputs)
```

- [ ] **Step 2: Диаризация** — `Diarisation/do_diarize.py`

Добавить импорт:
```python
from core import gpu_profiler
```
Обернуть инференс (строка 207 `self.embedding_session.run_with_iobinding(io_binding)`):
```python
            with gpu_profiler.profile_block("diar", batch=int(batch.shape[0])):
                self.embedding_session.run_with_iobinding(io_binding)
```

- [ ] **Step 3: Пунктуация** — `Punctuation/__init__.py`

Добавить импорт:
```python
from core import gpu_profiler
```
Обернуть `await loop.run_in_executor(...)` (строки 142-150). Было:
```python
        outputs = await loop.run_in_executor(
            None,
            lambda: self.session.run(None, {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                    } ))
```
Стало:
```python
        async with gpu_profiler.profile("punct", n=int(input_ids.shape[-1])):
            outputs = await loop.run_in_executor(
                None,
                lambda: self.session.run(None, {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                        } ))
```

- [ ] **Step 4: GigaAM** — `Recognizer/engine/stream_recognition.py`

Добавить импорт (рядом с `from utils.audio_buckets import pad_to_bucket`):
```python
from core import gpu_profiler
```
Обернуть вызов `recognizer.recognize(...)` в `_simple_recognise_sync`. Было:
```python
    return asdict(recognizer.recognize(samples, sample_rate=settings.BASE_SAMPLE_RATE))
```
Стало:
```python
    with gpu_profiler.profile_block("giga", samples=int(len(samples))):
        result = asdict(recognizer.recognize(samples, sample_rate=settings.BASE_SAMPLE_RATE))
    return result
```

- [ ] **Step 5: T-one** — `utils/tone_stream.py`

Добавить импорт (после `from config import settings`):
```python
from core import gpu_profiler
```
Обернуть тело `forward_async` (await run_in_executor):
```python
async def forward_async(executor, pipeline, samples, state, *, is_last: bool = False):
    loop = asyncio.get_running_loop()
    async with gpu_profiler.profile("tone", n=int(len(samples)), is_last=bool(is_last)):
        return await loop.run_in_executor(
            executor,
            functools.partial(pipeline.forward, samples, state, is_last=is_last),
        )
```
и `finalize_async`:
```python
async def finalize_async(executor, pipeline, state):
    loop = asyncio.get_running_loop()
    async with gpu_profiler.profile("tone", action="finalize"):
        return await loop.run_in_executor(executor, pipeline.finalize, state)
```

- [ ] **Step 6: Регистрация очереди + kind у T-one** — `api/v1/endpoints/asr_ws_tone.py`

Добавить импорт:
```python
from core import gpu_profiler
```
Call site (строка 63): передать `kind="tone"`:
```python
    if not await manager.connect(websocket, client_id, kind="tone"):
        return  # лимит соединений исчерпан, ConnectionManager уже закрыл сокет
```
После создания `audio_q` (строка с `audio_q: asyncio.Queue = asyncio.Queue()`) зарегистрировать очередь и снять в finally. Сразу после строки создания `audio_q`:
```python
    gpu_profiler.register_queue(audio_q)
```
В финальном `finally` обработчика (где `await manager.disconnect(client_id)`) — первой строкой:
```python
        gpu_profiler.unregister_queue(audio_q)
```

- [ ] **Step 7: kind у GigaAM** — `api/v1/endpoints/asr_ws.py`

Call site (строка 79): передать `kind="giga"`:
```python
    if not await manager.connect(websocket, client_id, kind="giga"):
```

- [ ] **Step 8: Проверка — компиляция + существующие тесты T-one**

Run: `venv/bin/python -m py_compile VoiceActivityDetector/do_vad.py Diarisation/do_diarize.py Punctuation/__init__.py Recognizer/engine/stream_recognition.py utils/tone_stream.py api/v1/endpoints/asr_ws_tone.py api/v1/endpoints/asr_ws.py`
Expected: без ошибок.

Run: `venv/bin/pytest tests/test_tone_offload.py tests/test_pad_buckets.py -q`
Expected: PASS (обёртки no-op при выкл, поведение не меняется).

- [ ] **Step 9: Коммит**

```bash
git add VoiceActivityDetector/do_vad.py Diarisation/do_diarize.py Punctuation/__init__.py Recognizer/engine/stream_recognition.py utils/tone_stream.py api/v1/endpoints/asr_ws_tone.py api/v1/endpoints/asr_ws.py
git commit -m "$(cat <<'EOF'
GPU-профиль: инструментация компонентов (vad/diar/punct/giga/tone) + kind + очередь

profile()-обёртки вокруг инференса (no-op при выкл), kind у connect (tone/giga),
регистрация audio_q T-one для бэклога.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Эндпоинт `/api/v1/admin/gpu-profile`

**Files:**
- Modify: `api/v1/endpoints/admin.py` (импорты + новый GET)

- [ ] **Step 1: Падающий тест**

Дописать в `tests/test_gpu_profiler.py`:

```python
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
```

- [ ] **Step 2: Запустить — упадёт**

Run: `venv/bin/pytest tests/test_gpu_profiler.py::test_gpu_profile_endpoint_enabled -q`
Expected: FAIL — `ImportError: cannot import name 'admin_gpu_profile'`.

- [ ] **Step 3: Реализовать эндпоинт в `api/v1/endpoints/admin.py`**

Добавить импорты вверху (если ещё нет):
```python
from fastapi import Request
from config import settings
from core import gpu_profiler
```
Добавить эндпоинт (после `admin_metrics`):
```python
@router.get("/gpu-profile")
async def admin_gpu_profile(request: Request, current_user: User = Depends(require_admin)):
    """Снимок GPU-профиля: память процесса, конкуренция по путям, бэклог, конфиг."""
    if not gpu_profiler.enabled():
        return {"enabled": False}
    mgr = getattr(request.app.state, "ws_manager", None)
    conns = mgr.counts_by_kind() if (mgr is not None and hasattr(mgr, "counts_by_kind")) else {}
    return {
        "enabled": True,
        "snapshot": gpu_profiler.snapshot(),
        "components": gpu_profiler.components(),
        "conns_by_path": conns,
        "tone_backlog": gpu_profiler.tone_backlog(),
        "config": {
            "provider": settings.PROVIDER,
            "stream_with_gpu": settings.STREAM_WITH_GPU,
            "tone_infer_workers": settings.TONE_INFER_WORKERS,
            "asr_pad_buckets_sec": settings.ASR_PAD_BUCKETS_SEC,
            "gpu_profile": settings.GPU_PROFILE,
        },
    }
```
Примечание: в тесте `current_user` и порядок параметров не важны (зовём функцию напрямую именованными аргументами). FastAPI порядок `request`/`Depends` допускает оба.

- [ ] **Step 4: Запустить — пройдёт**

Run: `venv/bin/pytest tests/test_gpu_profiler.py -q`
Expected: PASS (все тесты файла).

- [ ] **Step 5: Коммит**

```bash
git add api/v1/endpoints/admin.py tests/test_gpu_profiler.py
git commit -m "$(cat <<'EOF'
GPU-профиль: эндпоинт GET /api/v1/admin/gpu-profile (снимок+агрегаты+конфиг)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Standalone-монитор `tools/gpu_monitor.py`

**Files:**
- Create: `tools/gpu_monitor.py`
- Modify: `tests/test_gpu_profiler.py` (тест анализа)

- [ ] **Step 1: Падающий тест анализа**

Дописать в `tests/test_gpu_profiler.py`:

```python
def test_monitor_analyze_plateau_vs_growth():
    from tools.gpu_monitor import analyze
    plateau = analyze([100, 200, 300, 300, 300, 300])
    assert plateau["max"] == 300
    assert plateau["verdict"] == "plateau"
    growth = analyze([100, 200, 300, 400, 500, 600])
    assert growth["verdict"] == "growing"
```

- [ ] **Step 2: Запустить — упадёт**

Run: `venv/bin/pytest tests/test_gpu_profiler.py::test_monitor_analyze_plateau_vs_growth -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.gpu_monitor'`.

- [ ] **Step 3: Создать `tools/gpu_monitor.py`**

```python
# -*- coding: utf-8 -*-
"""
Standalone-монитор GPU-профиля.

Опрашивает /api/v1/admin/gpu-profile (URL+токен) раз в N сек, пишет CSV и
печатает таблицу; считает рост/плато по process_gpu_mib и корреляцию с
конкуренцией. Альтернатива — тейл logs/gpu_profile.jsonl (--jsonl).

Запуск:
  python tools/gpu_monitor.py --url http://127.0.0.1:49153 --token <JWT> --interval 2 --csv /tmp/gpu.csv
  python tools/gpu_monitor.py --jsonl logs/gpu_profile.jsonl
"""

import argparse
import json
import time
import urllib.request


def analyze(series) -> dict:
    """По ряду process_gpu_mib: max, дельта, вердикт plateau|growing|empty."""
    vals = [v for v in series if v is not None]
    if not vals:
        return {"max": None, "delta_tail": None, "verdict": "empty"}
    tail = vals[-min(4, len(vals)):]
    rising = all(b >= a for a, b in zip(tail, tail[1:])) and (tail[-1] - tail[0]) > 0
    verdict = "growing" if rising and len(tail) >= 3 else "plateau"
    return {"max": max(vals), "delta_tail": tail[-1] - tail[0], "verdict": verdict}


def _fetch(url: str, token: str) -> dict:
    req = urllib.request.Request(url.rstrip("/") + "/api/v1/admin/gpu-profile",
                                 headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--token", default="")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--csv", default="")
    ap.add_argument("--jsonl", default="")
    args = ap.parse_args()

    series = []
    csv_f = open(args.csv, "w", encoding="utf-8") if args.csv else None
    if csv_f:
        csv_f.write("ts,process_gpu_mib,gpu_used_mib,tone_backlog,conns_total\n")

    try:
        while True:
            if args.url:
                d = _fetch(args.url, args.token)
                if not d.get("enabled"):
                    print("GPU_PROFILE выключен на сервере"); return
                snap = d.get("snapshot", {})
                pm = snap.get("process_gpu_mib")
                used = snap.get("gpu_used_mib")
                backlog = d.get("tone_backlog")
                conns = (d.get("conns_by_path") or {}).get("total")
            else:
                # тейл JSONL: последняя sample-строка
                pm = used = backlog = conns = None
                try:
                    with open(args.jsonl, "r", encoding="utf-8") as f:
                        lines = f.readlines()[-200:]
                    for ln in reversed(lines):
                        o = json.loads(ln)
                        if o.get("kind") == "sample":
                            pm = o.get("process_gpu_mib"); used = o.get("gpu_used_mib")
                            backlog = o.get("tone_backlog")
                            conns = (o.get("conns_by_path") or {}).get("total")
                            break
                except Exception:
                    pass
            series.append(pm)
            a = analyze(series)
            print(f"proc={pm} used={used} backlog={backlog} conns={conns} | max={a['max']} verdict={a['verdict']}")
            if csv_f:
                csv_f.write(f"{time.time()},{pm},{used},{backlog},{conns}\n"); csv_f.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if csv_f:
            csv_f.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Запустить — пройдёт**

Run: `venv/bin/pytest tests/test_gpu_profiler.py -q`
Expected: PASS (все тесты файла).

- [ ] **Step 5: Коммит**

```bash
git add tools/gpu_monitor.py tests/test_gpu_profiler.py
git commit -m "$(cat <<'EOF'
GPU-профиль: standalone-монитор tools/gpu_monitor.py (опрос/тейл, CSV, плато/рост)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Живая проверка (на GPU)

- [ ] **Step 1: Поднять сервер с профилем**

Run (фоном): `HOST=127.0.0.1 PORT=49153 PROVIDER=CUDA STREAM_WITH_GPU=1 IS_PROD=0 GPU_PROFILE=1 venv/bin/python main.py`
Expected в логе: `GPU-профилировщик включён (сэмплер 1с -> logs/gpu_profile.jsonl)`.

- [ ] **Step 2: Дать нагрузку (12 поканально) и снять профиль**

Прогнать `python /tmp/ws_chan12.py /tmp/rec2_ch0.wav /tmp/rec2_ch1.wav 0.3`, затем:

Run: `tail -3 logs/gpu_profile.jsonl`
Expected: `sample`-строки с растущим `process_gpu_mib`, `conns_by_path`, `tone_backlog`; component-строки `giga`/`tone`.

- [ ] **Step 3: Эндпоинт**

Run: `curl -s -H "Authorization: Bearer <admin-jwt>" http://127.0.0.1:49153/api/v1/admin/gpu-profile`
Expected: JSON `enabled:true`, `snapshot.process_gpu_mib` число, `conns_by_path`, `components` с giga/tone, `config`.

- [ ] **Step 4: Монитор**

Run: `venv/bin/python tools/gpu_monitor.py --jsonl logs/gpu_profile.jsonl --interval 2`
Expected: строки `proc=… used=… backlog=… conns=… verdict=plateau|growing`.

---

## Self-Review (заполнено автором плана)

**Покрытие спека:**
- Гейт GPU_PROFILE → Task 1 (config) ✓
- Кэш-снимок + record/profile/profile_block + компоненты-агрегаты → Task 1 ✓
- Реестр очередей/бэклог → Task 1 (+ регистрация Task 5) ✓
- NVML-снимок (process GPU MiB) + устойчивость без GPU → Task 1/2 ✓
- counts_by_kind по путям → Task 3 ✓
- Фон-сэмплер + lifespan → Task 1 (loop) + Task 4 (wiring) ✓
- Инструментация vad/diar/punct/giga/tone → Task 5 ✓
- Эндпоинт → Task 6 ✓
- Standalone-монитор → Task 7 ✓
- Живой прогон → Task 8 ✓
- Вне области (побайтовая разбивка, БД-persist, probe) — не реализуем ✓

**Плейсхолдеры:** нет (весь код приведён).

**Согласованность имён:** `enabled()`, `record()`, `profile()`, `profile_block()`, `snapshot()`, `components()`, `register_queue()`/`unregister_queue()`/`tone_backlog()`, `read_gpu_snapshot()`, `gpu_sampler_loop()`, `counts_by_kind()`, `admin_gpu_profile`, `analyze()` — определены в ранних задачах и используются согласованно в поздних. `GPU_PROFILE`, `app.state.gpu_profile_task`, `logs/gpu_profile.jsonl` — одно имя везде. ✓
