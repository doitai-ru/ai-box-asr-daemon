# Вынос инференса T-one из event-loop — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Снять инференс T-one (`/api/v1/asr/ws-stream`) с event-loop'а в выделенный поток-исполнитель, чтобы под нагрузкой loop не голодал и uvicorn не ронял сокеты каскадом с 1011.

**Architecture:** Доказано (см. спек), что общий синглтон `StreamingCTCPipeline` потокобезопасен при beam-search/kenlm. Поэтому добавляем выделенный `ThreadPoolExecutor` (по умолчанию 1 воркер, параметр `TONE_INFER_WORKERS`), создаём его в lifespan, а в WS-обработчике вызываем `pipeline.forward()/finalize()` через `loop.run_in_executor(...)` (тонкие async-хелперы `forward_async`/`finalize_async`). Хрупкая логика прокидывания `state` и формирования фраз не меняется — только точка вызова.

**Tech Stack:** Python 3.12, FastAPI/Starlette WebSocket, asyncio, `concurrent.futures.ThreadPoolExecutor`, pydantic-settings, pytest 9 (тесты синхронные через `asyncio.run`, без зависимости от pytest-asyncio — её в venv нет).

**Спек:** `docs/superpowers/specs/2026-06-08-tone-streaming-offload-design.md`

**Команды:** интерпретатор — `venv/bin/python`; тесты — `venv/bin/pytest`. Запускаем **только новый файл теста** (`tests/test_tone_offload.py`): прочие тесты используют `@pytest.mark.asyncio`, а pytest-asyncio не установлен, поэтому общий прогон зашумлён.

---

## Структура файлов

- **Изменяем** `config.py` — добавляем настройку `TONE_INFER_WORKERS` в блок «T-one streaming settings».
- **Изменяем** `utils/tone_stream.py` — добавляем фабрику `make_tone_executor()` и async-хелперы `forward_async()`/`finalize_async()` (тонкие обёртки над `run_in_executor`). Это естественный дом: файл уже «хелперы для потокового распознавания через T-one».
- **Изменяем** `main.py` (lifespan) — создаём `app.state.tone_executor` на старте, закрываем на shutdown.
- **Изменяем** `api/v1/endpoints/asr_ws_tone.py` — вызовы `pipeline.forward()/finalize()` уводим через хелперы в executor.
- **Создаём** `tests/test_tone_offload.py` — тесты конфигурации, фабрики/хелперов и проводки обработчика.

---

## Task 1: Настройка `TONE_INFER_WORKERS`

**Files:**
- Create: `tests/test_tone_offload.py`
- Modify: `config.py` (блок «T-one streaming settings», после `STREAM_WITH_GPU`, ~строка 43)

- [ ] **Step 1: Написать падающий тест**

Создать `tests/test_tone_offload.py` с содержимым:

```python
# -*- coding: utf-8 -*-
"""Тесты выноса инференса T-one из event-loop (offload в выделенный executor)."""

import asyncio
import threading

import numpy as np
import pytest


# ── Task 1: конфигурация ────────────────────────────────────────────────────

def test_tone_infer_workers_default_is_one():
    from config import Settings
    assert Settings().TONE_INFER_WORKERS == 1


def test_tone_infer_workers_overridable():
    from config import Settings
    assert Settings(TONE_INFER_WORKERS=4).TONE_INFER_WORKERS == 4
```

- [ ] **Step 2: Запустить тест — убедиться, что падает**

Run: `venv/bin/pytest tests/test_tone_offload.py -v`
Expected: FAIL — `AttributeError`/`ValidationError`: у `Settings` нет поля `TONE_INFER_WORKERS`.

- [ ] **Step 3: Добавить настройку в `config.py`**

В блок «T-one streaming settings», сразу после поля `STREAM_WITH_GPU` (перед строкой `# HuggingFace Hub settings`), вставить:

```python
    # Число потоков-воркеров под инференс T-one (вынос инференса с event-loop'а
    # в отдельный исполнитель). 1 - безопасный дефолт: минимум конкуренции за GIL
    # с event-loop'ом, keepalive не голодает. При greedy-декодере можно поднять;
    # параллелизм beam-search всё равно ограничен GIL (честное масштабирование -
    # это мультипроцесс/процесс-на-карту, отдельный этап).
    TONE_INFER_WORKERS: int = 1
```

- [ ] **Step 4: Запустить тест — убедиться, что проходит**

Run: `venv/bin/pytest tests/test_tone_offload.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Коммит**

```bash
git add config.py tests/test_tone_offload.py
git commit -m "$(cat <<'EOF'
T-one: настройка TONE_INFER_WORKERS (число воркеров offload-исполнителя)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Фабрика исполнителя и async-хелперы offload

**Files:**
- Modify: `tests/test_tone_offload.py` (дописать тесты)
- Modify: `utils/tone_stream.py` (добавить импорты + 3 функции)

- [ ] **Step 1: Написать падающие тесты**

Дописать в конец `tests/test_tone_offload.py`:

```python
# ── Task 2: фабрика executor'а и async-хелперы offload ──────────────────────

class _FakePipeline:
    """Фейковый пайплайн: фиксирует, на каком потоке его вызвали."""

    def __init__(self):
        self.calls = []

    def forward(self, samples, state, *, is_last=False):
        self.calls.append((threading.current_thread().name, is_last))
        return ([f"phrase:{is_last}"], "state-after-forward")

    def finalize(self, state):
        self.calls.append((threading.current_thread().name, "finalize"))
        return (["final"], "state-after-finalize")


def test_make_tone_executor_threads_are_named_tone():
    from utils.tone_stream import make_tone_executor
    ex = make_tone_executor(2)
    try:
        name = ex.submit(lambda: threading.current_thread().name).result()
    finally:
        ex.shutdown(wait=True)
    assert name.startswith("tone")


def test_make_tone_executor_clamps_to_at_least_one():
    from utils.tone_stream import make_tone_executor
    ex = make_tone_executor(0)  # некорректное значение не должно ломать executor
    try:
        result = ex.submit(lambda: 42).result()
    finally:
        ex.shutdown(wait=True)
    assert result == 42


def test_forward_async_runs_off_caller_thread_on_tone_executor():
    from utils.tone_stream import forward_async, make_tone_executor
    pipe = _FakePipeline()
    ex = make_tone_executor(1)
    try:
        phrases, state = asyncio.run(
            forward_async(ex, pipe, np.zeros(4, dtype=np.int32), None)
        )
    finally:
        ex.shutdown(wait=True)
    assert phrases == ["phrase:False"]
    assert state == "state-after-forward"
    thread_name, is_last = pipe.calls[0]
    assert thread_name.startswith("tone")                        # ушло в выделенный executor
    assert thread_name != threading.current_thread().name        # не на вызывающем потоке
    assert is_last is False


def test_forward_async_passes_is_last_true():
    from utils.tone_stream import forward_async, make_tone_executor
    pipe = _FakePipeline()
    ex = make_tone_executor(1)
    try:
        phrases, _ = asyncio.run(
            forward_async(ex, pipe, np.zeros(4, dtype=np.int32), None, is_last=True)
        )
    finally:
        ex.shutdown(wait=True)
    assert phrases == ["phrase:True"]
    assert pipe.calls[0][1] is True


def test_finalize_async_runs_off_caller_thread():
    from utils.tone_stream import finalize_async, make_tone_executor
    pipe = _FakePipeline()
    ex = make_tone_executor(1)
    try:
        phrases, state = asyncio.run(finalize_async(ex, pipe, "s0"))
    finally:
        ex.shutdown(wait=True)
    assert phrases == ["final"]
    assert state == "state-after-finalize"
    assert pipe.calls[0][0].startswith("tone")
```

- [ ] **Step 2: Запустить тесты — убедиться, что падают**

Run: `venv/bin/pytest tests/test_tone_offload.py -v`
Expected: FAIL — `ImportError: cannot import name 'make_tone_executor'` (и `forward_async`/`finalize_async`).

- [ ] **Step 3: Реализовать фабрику и хелперы в `utils/tone_stream.py`**

В начало файла, к существующим импортам (после `import numpy as np` / `import soxr`), добавить:

```python
import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
```

В конец файла (после `phrase_to_data`) добавить:

```python
def make_tone_executor(max_workers: int) -> ThreadPoolExecutor:
    """
    Выделенный пул под инференс T-one (вне event-loop'а).

    Имя потоков 'tone*' — чтобы offload было видно в логах/трейсах. Значение
    клампится к >= 1 (на случай некорректного TONE_INFER_WORKERS из окружения).
    """
    return ThreadPoolExecutor(
        max_workers=max(1, int(max_workers)),
        thread_name_prefix="tone",
    )


async def forward_async(executor, pipeline, samples, state, *, is_last: bool = False):
    """
    Выполняет pipeline.forward(samples, state, is_last=...) в выделенном executor'е.

    Снимает синхронный инференс с event-loop'а. Вызовы на один коннект await'ятся
    по очереди — порядок прокидывания state сохраняется.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        executor,
        functools.partial(pipeline.forward, samples, state, is_last=is_last),
    )


async def finalize_async(executor, pipeline, state):
    """Выполняет pipeline.finalize(state) в выделенном executor'е (вне event-loop'а)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, pipeline.finalize, state)
```

- [ ] **Step 4: Запустить тесты — убедиться, что проходят**

Run: `venv/bin/pytest tests/test_tone_offload.py -v`
Expected: PASS (7 passed: 2 из Task 1 + 5 новых).

- [ ] **Step 5: Коммит**

```bash
git add utils/tone_stream.py tests/test_tone_offload.py
git commit -m "$(cat <<'EOF'
T-one: фабрика executor'а и async-хелперы offload инференса

make_tone_executor + forward_async/finalize_async — тонкие обёртки
над run_in_executor для выноса pipeline.forward/finalize с event-loop'а.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Создание/закрытие `tone_executor` в lifespan (`main.py`)

**Files:**
- Modify: `main.py` (импорт + блок lifespan: создание после прогрева T-one, закрытие на shutdown)

Примечание: lifespan грузит реальные модели, поэтому юнит-тестом не покрываем — проверяем компиляцию и импорт-смоук. Поведение offload'а проверено в Task 2 (хелперы) и будет проверено в Task 4 (проводка обработчика).

- [ ] **Step 1: Добавить импорт фабрики**

В `main.py`, рядом с прочими импортами из `utils` (после `from utils.pre_start_init import paths`), добавить:

```python
from utils.tone_stream import make_tone_executor
```

- [ ] **Step 2: Создать executor на старте**

В функции `lifespan`, сразу ПОСЛЕ блока `try/except`, который инициализирует и прогревает `app.state.tone_pipeline` (заканчивается строкой `logger.error("Не удалось инициализировать T-one на старте: %s", exc)`), и ПЕРЕД блоком `if settings.DO_LOCAL_FILE_RECOGNITIONS:`, вставить:

```python
    # Выделенный исполнитель под инференс T-one (вне event-loop'а): поток /ws-stream
    # не должен голодить loop, иначе uvicorn рвёт сокеты каскадом (keepalive 1011).
    app.state.tone_executor = make_tone_executor(settings.TONE_INFER_WORKERS)
    logger.info("T-one executor создан (workers=%s)", settings.TONE_INFER_WORKERS)
```

- [ ] **Step 3: Закрыть executor на shutdown**

В том же `lifespan`, в секции завершения, сразу ПОСЛЕ строки `await app.state.ws_manager.disconnect_all()` (внутри `if hasattr(app.state, "ws_manager"):`), добавить:

```python
    # Останавливаем offload-исполнитель T-one
    if hasattr(app.state, "tone_executor"):
        app.state.tone_executor.shutdown(wait=True)
        logger.debug("T-one executor остановлен")
```

- [ ] **Step 4: Проверить компиляцию и импорт**

Run: `venv/bin/python -m py_compile main.py utils/tone_stream.py && venv/bin/python -c "import main; print('import main OK')"`
Expected: вывод заканчивается `import main OK` (предупреждения WARNING:root про CORS/SECRET_KEY — норма, не ошибки).

- [ ] **Step 5: Коммит**

```bash
git add main.py
git commit -m "$(cat <<'EOF'
T-one: создаём/закрываем tone_executor в lifespan

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Проводка обработчика `/ws-stream` через executor

**Files:**
- Modify: `tests/test_tone_offload.py` (дописать тест проводки обработчика)
- Modify: `api/v1/endpoints/asr_ws_tone.py` (импорт хелперов, получение executor, 3 точки вызова)

- [ ] **Step 1: Написать падающий тест проводки обработчика**

Дописать в конец `tests/test_tone_offload.py`:

```python
# ── Task 4: проводка WS-обработчика через executor ──────────────────────────

def test_ws_stream_offloads_inference_via_executor(monkeypatch):
    """Обработчик зовёт forward_async/finalize_async с app.state.tone_executor."""
    import api.v1.endpoints.asr_ws_tone as mod

    class Evt:
        def __init__(self, kind, audio=b"", sample_rate=8000, channel_name="Null"):
            self.kind = kind
            self.audio = audio
            self.sample_rate = sample_rate
            self.channel_name = channel_name

    # config(8 кГц) -> один полный кадр T-one -> disconnect
    frame = b"\x00" * (mod.settings.TONE_CHUNK_SAMPLES * 2)
    events = [Evt("config", sample_rate=8000), Evt("audio", audio=frame), Evt("disconnect")]

    class FakeWS:
        def __init__(self, app):
            self.app = app
            self._it = iter(events)

        async def receive(self):
            return next(self._it)

    class FakeManager:
        def __init__(self):
            self.sent = []

        async def connect(self, ws, cid):
            return True

        async def send_message(self, cid, msg):
            self.sent.append(msg)

        async def disconnect(self, cid):
            pass

    sentinel_executor = object()
    sentinel_pipeline = object()

    app = type("App", (), {})()
    app.state = type("State", (), {})()
    app.state.tone_pipeline = sentinel_pipeline
    app.state.tone_executor = sentinel_executor
    app.state.ws_manager = FakeManager()

    forward_calls = []
    finalize_calls = []

    async def fake_forward_async(executor, pipeline, samples, state, *, is_last=False):
        forward_calls.append((executor, pipeline, is_last))
        return ([], "state1")

    async def fake_finalize_async(executor, pipeline, state):
        finalize_calls.append((executor, pipeline))
        return ([], "state2")

    monkeypatch.setattr(mod, "detect", lambda m: m)              # полученный Evt и есть событие
    monkeypatch.setattr(mod, "forward_async", fake_forward_async)
    monkeypatch.setattr(mod, "finalize_async", fake_finalize_async)

    asyncio.run(mod.websocket_tone_stream(FakeWS(app)))

    assert len(forward_calls) == 1                               # один кадр -> один forward
    assert forward_calls[0][0] is sentinel_executor             # использован app.state.tone_executor
    assert forward_calls[0][1] is sentinel_pipeline
    assert forward_calls[0][2] is False
    assert len(finalize_calls) == 1                             # финализация тоже offload'ится
    assert finalize_calls[0][0] is sentinel_executor
```

- [ ] **Step 2: Запустить тест — убедиться, что падает**

Run: `venv/bin/pytest tests/test_tone_offload.py::test_ws_stream_offloads_inference_via_executor -v`
Expected: FAIL — `AttributeError`: в модуле `asr_ws_tone` нет `forward_async` (его ещё не импортировали), либо обработчик не обращается к `tone_executor`/не вызывает хелперы.

- [ ] **Step 3: Реализовать проводку в `api/v1/endpoints/asr_ws_tone.py`**

3a. Расширить импорт из `utils.tone_stream` (строка 33) — добавить `forward_async, finalize_async`:

```python
from utils.tone_stream import take_frames, flush_tail, phrase_to_data, StreamResampler, forward_async, finalize_async
```

3b. Получить executor: сразу ПОСЛЕ строки
`pipeline = getattr(websocket.app.state, "tone_pipeline", None) or get_tone_pipeline()`
добавить строку:

```python
    executor = websocket.app.state.tone_executor
```

3c. В аудио-цикле заменить синхронный вызов (было `phrases, state = pipeline.forward(samples, state)`):

```python
                    for samples in take_frames(buf):
                        phrases, state = await forward_async(executor, pipeline, samples, state)
                        for phrase in phrases:
                            if phrase.text:
                                await manager.send_message(client_id, _result_message(phrase, channel_name))
```

3d. В финализации заменить два синхронных вызова. Было:

```python
            if tail is not None:
                phrases, state = pipeline.forward(tail, state, is_last=True)
                final_phrases.extend(phrases)
            fin_phrases, state = pipeline.finalize(state)
```

Стало:

```python
            if tail is not None:
                phrases, state = await forward_async(executor, pipeline, tail, state, is_last=True)
                final_phrases.extend(phrases)
            fin_phrases, state = await finalize_async(executor, pipeline, state)
```

- [ ] **Step 4: Запустить тест — убедиться, что проходит**

Run: `venv/bin/pytest tests/test_tone_offload.py -v`
Expected: PASS (8 passed — все тесты файла).

- [ ] **Step 5: Коммит**

```bash
git add api/v1/endpoints/asr_ws_tone.py tests/test_tone_offload.py
git commit -m "$(cat <<'EOF'
T-one: вынос инференса /ws-stream в executor (фикс голодания event-loop)

pipeline.forward/finalize теперь идут через forward_async/finalize_async
в app.state.tone_executor — loop свободен отвечать на ping/pong,
каскадные 1011 устранены. Beam-search/kenlm потокобезопасен (см. спек).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Финальная проверка

- [ ] **Прогон всех новых тестов**

Run: `venv/bin/pytest tests/test_tone_offload.py -v`
Expected: PASS (8 passed).

- [ ] **Импорт-смоук приложения**

Run: `venv/bin/python -c "import main; print('OK')"`
Expected: заканчивается `OK`.

- [ ] **(Опционально, требует GPU/моделей) Ручная проверка под нагрузкой**

Поднять сервис, дать параллельную нагрузку на `/api/v1/asr/ws-stream` и убедиться, что
офлайн-ручка `/ws` и другие потоки больше не падают каскадом с 1011, а результаты
распознавания совпадают с прежними (beam-search по умолчанию).

---

## Self-Review (заполнено автором плана)

**Покрытие спека:**
- Параметр `TONE_INFER_WORKERS` (дефолт 1) → Task 1. ✓
- Выделенный executor, создание/закрытие в lifespan → Task 3 (+ фабрика в Task 2). ✓
- Offload `forward`/`finalize` в `asr_ws_tone.py` → Task 4 (через хелперы из Task 2). ✓
- Не используем общий `asyncio.to_thread` → отдельный executor (`make_tone_executor`). ✓
- Критерий «результаты идентичны» → проверяется ручным шагом (детерминизм beam-search не меняется, меняется только поток исполнения). ✓
- Out of scope (multi-GPU, nginx, ws-ping) → в плане не трогаются. ✓

**Плейсхолдеры:** нет (весь код приведён целиком).

**Согласованность имён/сигнатур:** `make_tone_executor(max_workers)`, `forward_async(executor, pipeline, samples, state, *, is_last=False)`, `finalize_async(executor, pipeline, state)` — одинаково определены (Task 2) и используются (Task 3 lifespan, Task 4 handler, тесты). `app.state.tone_executor` — одно имя везде. ✓
