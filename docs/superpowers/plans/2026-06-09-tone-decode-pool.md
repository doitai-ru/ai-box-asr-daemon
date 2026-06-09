# T-one decode-пул — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Раскидать kenlm beam-search декод T-one по `ProcessPoolExecutor` (раздельные GIL → параллель по ядрам CPU), оставив GPU-акустику и нарезку фраз в основном процессе; выключаемо через `TONE_DECODE_PROCS`.

**Architecture:** На коннект: стейдж A (`model.forward`+`splitter.forward` в tone_executor, без декода) → фразы; стейдж B (kenlm-декод каждой фразы) → в пул процессов. Одна GPU-модель, kenlm mmap-шарится. `TONE_DECODE_PROCS=0` (дефолт) — текущий in-process путь (фолбэк).

**Tech Stack:** Python 3.12, `tone`/pyctcdecode/kenlm, `concurrent.futures.ProcessPoolExecutor`, asyncio, pytest 9 (тесты синхронные через `asyncio.run`).

**Спек:** `docs/superpowers/specs/2026-06-09-tone-decode-pool-design.md`

**Команды:** интерпретатор `venv/bin/python`, тесты `venv/bin/pytest tests/test_tone_decode_pool.py -v`.

---

## Структура файлов

- **Изменяем** `config.py` — `TONE_DECODE_PROCS`, `TONE_BEAM_WIDTH`.
- **Изменяем** `Recognizer/tone_engine.py` — резолвер пути kenlm, фабрика пула, worker-функция декода.
- **Изменяем** `utils/tone_stream.py` — `forward_split`/`forward_split_async`, `decode_async`.
- **Изменяем** `api/v1/endpoints/asr_ws_tone.py` — inferer: путь через пул или фолбэк.
- **Изменяем** `main.py` — создание/закрытие пула в lifespan.
- **Создаём** `tests/test_tone_decode_pool.py`.

---

## Task 1: config TONE_DECODE_PROCS + TONE_BEAM_WIDTH

**Files:**
- Create: `tests/test_tone_decode_pool.py`
- Modify: `config.py` (после `TONE_INFER_WORKERS`)

- [ ] **Step 1: Падающий тест**

Создать `tests/test_tone_decode_pool.py`:

```python
# -*- coding: utf-8 -*-
"""Тесты T-one decode-пула."""

import asyncio
import concurrent.futures as cf

import numpy as np
import pytest


def test_config_decode_pool_defaults():
    from config import Settings
    s = Settings()
    assert s.TONE_DECODE_PROCS == 0      # дефолт: in-process (фолбэк)
    assert s.TONE_BEAM_WIDTH == 200      # как в библиотеке tone


def test_config_decode_pool_overridable():
    from config import Settings
    s = Settings(TONE_DECODE_PROCS=4, TONE_BEAM_WIDTH=50)
    assert s.TONE_DECODE_PROCS == 4
    assert s.TONE_BEAM_WIDTH == 50
```

- [ ] **Step 2: Запустить — упадёт**

Run: `venv/bin/pytest tests/test_tone_decode_pool.py -q`
Expected: FAIL — у `Settings` нет `TONE_DECODE_PROCS`/`TONE_BEAM_WIDTH`.

- [ ] **Step 3: Добавить в `config.py`** (после строки `TONE_INFER_WORKERS: int = 1`):

```python
    # Декод T-one (kenlm beam-search) — CPU+GIL-bound. Раскидать по процессам:
    # 0 = декод в основном процессе (текущий путь, фолбэк); N>0 = пул из N процессов
    # (раздельные GIL -> параллель по ядрам). kenlm mmap-шарится между воркерами.
    TONE_DECODE_PROCS: int = 0
    # Ширина луча beam-search (рычаг стоимости декода; в библиотеке tone захардкожено 200).
    TONE_BEAM_WIDTH: int = 200
```

- [ ] **Step 4: Запустить — пройдёт**

Run: `venv/bin/pytest tests/test_tone_decode_pool.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Коммит**

```bash
git add config.py tests/test_tone_decode_pool.py
git commit -m "$(cat <<'EOF'
T-one: config TONE_DECODE_PROCS (0=in-process) + TONE_BEAM_WIDTH

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: kenlm-резолвер + фабрика пула + worker-декод (tone_engine)

**Files:**
- Modify: `Recognizer/tone_engine.py`
- Modify: `tests/test_tone_decode_pool.py`

- [ ] **Step 1: Падающий тест**

Дописать в `tests/test_tone_decode_pool.py`:

```python
def test_kenlm_path_resolver(tmp_path, monkeypatch):
    from Recognizer import tone_engine
    # модель tone в {HF_HOME}/tone/kenlm.bin
    d = tmp_path / "tone"; d.mkdir()
    (d / "kenlm.bin").write_bytes(b"x")
    monkeypatch.setattr("config.settings.HF_HOME", str(tmp_path))
    assert tone_engine._kenlm_path() == str(d / "kenlm.bin")


def test_make_decode_pool_gate(monkeypatch):
    from Recognizer import tone_engine
    monkeypatch.setattr("config.settings.TONE_DECODE_PROCS", 0)
    assert tone_engine.make_decode_pool() is None     # 0 -> нет пула (фолбэк)
```

- [ ] **Step 2: Запустить — упадёт**

Run: `venv/bin/pytest tests/test_tone_decode_pool.py::test_kenlm_path_resolver tests/test_tone_decode_pool.py::test_make_decode_pool_gate -q`
Expected: FAIL — нет `_kenlm_path`/`make_decode_pool`.

- [ ] **Step 3: Добавить в `Recognizer/tone_engine.py`** (в конец файла):

```python
import glob
from concurrent.futures import ProcessPoolExecutor

_decode_pool = None
_worker_decoder = None  # в каждом процессе пула


def _kenlm_path() -> str | None:
    """Путь до kenlm.bin: {HF_HOME}/tone/kenlm.bin, иначе из HF-кэша hub."""
    direct = os.path.join(_model_dir(), "kenlm.bin")
    if os.path.exists(direct):
        return direct
    hits = glob.glob(os.path.join(settings.HF_HOME, "hub", "models--t-tech--T-one",
                                  "snapshots", "*", "kenlm.bin"))
    return hits[0] if hits else None


def _decode_pool_init(kenlm_path: str) -> None:
    """initializer воркера: грузит beam-search декодер один раз (kenlm mmap-шарится)."""
    global _worker_decoder
    from tone.decoder import BeamSearchCTCDecoder
    _worker_decoder = BeamSearchCTCDecoder.from_local(kenlm_path)


def _decode_worker(logprobs, beam_width: int) -> str:
    """Декод одной фразы в процессе пула (configurable beam_width поверх хардкода tone)."""
    return _worker_decoder._decoder.decode(logprobs, beam_width=beam_width)


def make_decode_pool():
    """ProcessPoolExecutor под декод (или None при TONE_DECODE_PROCS<=0 / отсутствии kenlm)."""
    procs = int(getattr(settings, "TONE_DECODE_PROCS", 0) or 0)
    if procs <= 0:
        return None
    kpath = _kenlm_path()
    if not kpath:
        logger.warning("decode-пул выключен: kenlm.bin не найден")
        return None
    logger.info("T-one decode-пул: %s процессов (kenlm=%s)", procs, kpath)
    return ProcessPoolExecutor(max_workers=procs, initializer=_decode_pool_init, initargs=(kpath,))


def get_decode_pool():
    """Ленивый синглтон decode-пула."""
    global _decode_pool
    if _decode_pool is None:
        _decode_pool = make_decode_pool()
    return _decode_pool
```

(Файл уже импортирует `os`, `settings`, `logger`, имеет `_model_dir()`.)

- [ ] **Step 4: Запустить — пройдёт**

Run: `venv/bin/pytest tests/test_tone_decode_pool.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Коммит**

```bash
git add Recognizer/tone_engine.py tests/test_tone_decode_pool.py
git commit -m "$(cat <<'EOF'
T-one: фабрика decode-пула + kenlm-резолвер + worker (tone_engine)

ProcessPoolExecutor, воркер грузит beam-search декодер (kenlm mmap), gate по
TONE_DECODE_PROCS. _decode_worker зовёт decode с configurable beam_width.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: forward_split + decode_async (tone_stream)

**Files:**
- Modify: `utils/tone_stream.py`
- Modify: `tests/test_tone_decode_pool.py`

- [ ] **Step 1: Падающий тест**

Дописать в `tests/test_tone_decode_pool.py`:

```python
class _FakeLPPhrase:
    def __init__(self, logprobs, sf, ef):
        self.logprobs = logprobs; self.start_frame = sf; self.end_frame = ef


class _FakeModel:
    def forward(self, audio_chunk, model_state):
        return np.zeros((1, 4, 35), dtype=np.float32), "model_state2"


class _FakeSplitter:
    def forward(self, logprobs, logprob_state, is_last=False):
        ph = _FakeLPPhrase(np.zeros((5, 35), dtype=np.float32), 10, 20)
        return [ph], "splitter_state2"


class _FakePipeline:
    PADDING = 2400
    def __init__(self):
        self.model = _FakeModel(); self.logprob_splitter = _FakeSplitter()
        self.decoder = self  # чтобы поймать, если decode позвали в стейдже A
        self.decode_called = False
    def forward(self, logprobs):  # decoder.forward — НЕ должен вызываться
        self.decode_called = True; return "X"


def test_forward_split_no_decode_and_state(monkeypatch):
    from utils import tone_stream
    from concurrent.futures import ThreadPoolExecutor
    pipe = _FakePipeline()
    ex = ThreadPoolExecutor(1)
    try:
        phrases, state = asyncio.run(
            tone_stream.forward_split_async(ex, pipe, np.zeros(2400, dtype=np.int32), None, False))
    finally:
        ex.shutdown(wait=True)
    assert pipe.decode_called is False              # декод НЕ в стейдже A
    assert state == ("model_state2", "splitter_state2")
    assert len(phrases) == 1
    lp, start, end = phrases[0]
    assert lp.shape == (5, 35)                       # logprobs фразы для отложенного декода
    assert isinstance(start, float) and end >= start


def test_decode_async_via_pool(monkeypatch):
    from utils import tone_stream
    from Recognizer import tone_engine
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setattr(tone_engine, "_decode_worker", lambda lp, bw: f"text:{lp.shape[0]}:{bw}")
    ex = ThreadPoolExecutor(2)
    try:
        txt = asyncio.run(tone_stream.decode_async(ex, np.zeros((7, 35), dtype=np.float32), 50))
    finally:
        ex.shutdown(wait=True)
    assert txt == "text:7:50"
```

- [ ] **Step 2: Запустить — упадёт**

Run: `venv/bin/pytest tests/test_tone_decode_pool.py::test_forward_split_no_decode_and_state tests/test_tone_decode_pool.py::test_decode_async_via_pool -q`
Expected: FAIL — нет `forward_split_async`/`decode_async`.

- [ ] **Step 3: Добавить в `utils/tone_stream.py`** (после `finalize_async`):

```python
def _forward_split(pipeline, samples, state, is_last):
    """Стейдж A: model.forward + splitter.forward БЕЗ декода. Копия pipeline.forward
    минус decoder.forward — возвращает (логпробы фразы, start, end) + новый state."""
    from tone.onnx_wrapper import StreamingCTCModel
    frame_size, time_bias = StreamingCTCModel.FRAME_SIZE, StreamingCTCModel.MEAN_TIME_BIAS
    padding, sr = pipeline.PADDING, StreamingCTCModel.SAMPLE_RATE

    model_state = state[0] if state is not None else None
    logprob_state = state[1] if state is not None else None

    logprobs, model_state_next = pipeline.model.forward(samples[None, :, None], model_state)
    logprob_phrases, logprob_state_next = pipeline.logprob_splitter.forward(
        logprobs[0], logprob_state, is_last=is_last)

    out = []
    for lp in logprob_phrases:
        start = max(0.0, round(lp.start_frame * frame_size - time_bias - padding / sr, 2))
        end = max(start, round(lp.end_frame * frame_size - time_bias - padding / sr, 2))
        out.append((lp.logprobs, start, end))
    return out, (model_state_next, logprob_state_next)


async def forward_split_async(executor, pipeline, samples, state, is_last: bool = False):
    """Стейдж A в tone_executor (GPU-акустика + нарезка, без декода)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        executor, functools.partial(_forward_split, pipeline, samples, state, is_last))


async def decode_async(pool, logprobs, beam_width: int):
    """Стейдж B: декод одной фразы в process-пуле (или thread-пуле в тестах)."""
    import Recognizer.tone_engine as te  # ссылка на воркер во время вызова (picklable + патчится)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(pool, te._decode_worker, logprobs, beam_width)
```

- [ ] **Step 4: Запустить — пройдёт**

Run: `venv/bin/pytest tests/test_tone_decode_pool.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Коммит**

```bash
git add utils/tone_stream.py tests/test_tone_decode_pool.py
git commit -m "$(cat <<'EOF'
T-one: forward_split (стейдж A, без декода) + decode_async (стейдж B в пул)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: inferer через пул (asr_ws_tone)

**Files:**
- Modify: `api/v1/endpoints/asr_ws_tone.py`

Примечание: путь через пул покрываем компиляцией + существующими тестами (фолбэк-путь
не меняется) и живым прогоном (Task 6). Логика: при наличии пула — `forward_split` +
`decode_async` на каждую фразу (порядок сохраняем), иначе — текущий `forward_async`.

- [ ] **Step 1: Импорты и получение пула**

В `api/v1/endpoints/asr_ws_tone.py` расширить импорт из `utils.tone_stream`:
```python
from utils.tone_stream import (take_frames, flush_tail, phrase_to_data, StreamResampler,
                               forward_async, finalize_async, forward_split_async, decode_async)
```
Добавить импорт:
```python
from config import settings as _settings  # для TONE_BEAM_WIDTH (settings уже импортирован как settings)
```
(если `settings` уже импортирован — используем его, отдельный импорт не нужен.)

Сразу после `executor = websocket.app.state.tone_executor` добавить:
```python
    decode_pool = getattr(websocket.app.state, "tone_decode_pool", None)
    beam_width = int(settings.TONE_BEAM_WIDTH)
```

- [ ] **Step 2: Хелпер сборки результата фразы из (logprobs,start,end)**

В `api/v1/endpoints/asr_ws_tone.py` рядом с `_result_message` добавить класс-обёртку фразы
(чтобы `_result_message`/`phrase_to_data` получили объект с `.text/.start_time/.end_time`):
```python
class _Phrase:
    __slots__ = ("text", "start_time", "end_time")
    def __init__(self, text, start_time, end_time):
        self.text = text; self.start_time = start_time; self.end_time = end_time
```

- [ ] **Step 3: Ветка пула в аудио-цикле inferer'а**

В `inferer()` заменить тело обработки кадра. Было:
```python
            samples, is_last = item
            try:
                phrases, state = await forward_async(executor, pipeline, samples, state, is_last=is_last)
            except Exception as exc:
                logger.error("[tone] recognize error %s (%s): %s", ctx["channel"], client_id, exc)
                continue
            if is_last:
                final_phrases.extend(phrases)
            else:
                for phrase in phrases:
                    if phrase.text:
                        await send(_result_message(phrase, ctx["channel"]))
```
Стало:
```python
            samples, is_last = item
            try:
                if decode_pool is not None:
                    raw, state = await forward_split_async(executor, pipeline, samples, state, is_last)
                    phrases = []
                    for logprobs, start, end in raw:                       # порядок сохраняем
                        text = await decode_async(decode_pool, logprobs, beam_width)
                        phrases.append(_Phrase(text, start, end))
                else:
                    phrases, state = await forward_async(executor, pipeline, samples, state, is_last=is_last)
            except Exception as exc:
                logger.error("[tone] recognize error %s (%s): %s", ctx["channel"], client_id, exc)
                continue
            if is_last:
                final_phrases.extend(phrases)
            else:
                for phrase in phrases:
                    if phrase.text:
                        await send(_result_message(phrase, ctx["channel"]))
```

- [ ] **Step 4: Финализация через пул**

В блоке финализации `inferer()` заменить:
```python
        try:
            fin_phrases, state = await finalize_async(executor, pipeline, state)
            final_phrases.extend(fin_phrases)
        except Exception as exc:
            logger.error("[tone] finalize error %s (%s): %s", ctx["channel"], client_id, exc)
```
на:
```python
        try:
            if decode_pool is not None:
                raw, state = await forward_split_async(
                    executor, pipeline, np.zeros(settings.TONE_CHUNK_SAMPLES, dtype=np.int32), state, True)
                for logprobs, start, end in raw:
                    text = await decode_async(decode_pool, logprobs, beam_width)
                    final_phrases.append(_Phrase(text, start, end))
            else:
                fin_phrases, state = await finalize_async(executor, pipeline, state)
                final_phrases.extend(fin_phrases)
        except Exception as exc:
            logger.error("[tone] finalize error %s (%s): %s", ctx["channel"], client_id, exc)
```
Добавить импорт numpy вверху файла, если его нет:
```python
import numpy as np
```
(финализация в библиотеке = `forward(zeros, state, is_last=True)` затем `finalize` — для пула
эквивалентно forward_split последнего кадра тишины; добивочные фразы декодятся в пуле.)

- [ ] **Step 5: Проверка — компиляция + существующие тесты**

Run: `venv/bin/python -m py_compile api/v1/endpoints/asr_ws_tone.py`
Expected: без ошибок.

Run: `venv/bin/pytest tests/test_tone_offload.py tests/test_tone_decode_pool.py -q`
Expected: PASS (фолбэк-путь не сломан: `tone_decode_pool` отсутствует в фейк-app → `None` → старый путь).

- [ ] **Step 6: Коммит**

```bash
git add api/v1/endpoints/asr_ws_tone.py
git commit -m "$(cat <<'EOF'
T-one /ws-stream: inferer через decode-пул (forward_split + decode в пул), фолбэк

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: пул в lifespan (main.py)

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Создание пула после tone_executor**

В `main.py`, сразу ПОСЛЕ строки `app.state.tone_executor = make_tone_executor(...)` добавить:
```python
    from Recognizer.tone_engine import make_decode_pool
    app.state.tone_decode_pool = make_decode_pool()
    if app.state.tone_decode_pool is not None:
        logger.info("T-one decode-пул создан (procs=%s)", settings.TONE_DECODE_PROCS)
```

- [ ] **Step 2: Закрытие на shutdown**

В секции shutdown, рядом с `app.state.tone_executor.shutdown(...)`, добавить:
```python
    if getattr(app.state, "tone_decode_pool", None) is not None:
        app.state.tone_decode_pool.shutdown(wait=False)
```

- [ ] **Step 3: Проверка**

Run: `venv/bin/python -m py_compile main.py && venv/bin/python -c "import main; print('OK')"`
Expected: заканчивается `OK` (при `TONE_DECODE_PROCS=0` пул не создаётся — `None`).

- [ ] **Step 4: Коммит**

```bash
git add main.py
git commit -m "$(cat <<'EOF'
T-one: создание/закрытие decode-пула в lifespan (при TONE_DECODE_PROCS>0)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: живая проверка (на GPU)

- [ ] **Step 1: Поднять с пулом**

Run (фоном): `HOST=127.0.0.1 PORT=49153 PROVIDER=CUDA STREAM_WITH_GPU=1 IS_PROD=0 GPU_PROFILE=1 TONE_DECODE_PROCS=4 venv/bin/python main.py`
Expected в логе: `T-one decode-пул создан (procs=4)`.

- [ ] **Step 2: kenlm mmap-шарится (RSS пула)**

Run: `for p in $(pgrep -f main.py); do ps -o pid,rss=,comm -p $p; done`
Expected: суммарный RSS воркеров пула НЕ растёт как 4×5.5 ГБ (kenlm mmap-шарится; видно по `free -h` shared/buff).

- [ ] **Step 3: 12 поканальных стримов — бэклог не копится**

Прогнать `python /tmp/ws_chan12.py /tmp/rec2_ch0.wav /tmp/rec2_ch1.wav 0.3`, параллельно:
Run: `tail -f logs/gpu_profile.jsonl | grep -o '"tone_backlog": [0-9]*'`
Expected: при `TONE_DECODE_PROCS=4` бэклог под 12 конкурентными растёт кратно медленнее/не растёт (сравнить с прогоном `TONE_DECODE_PROCS=0`).

- [ ] **Step 4: текст/таймкоды идентичны фолбэку**

Прогнать `python /tmp/ws_prod.py /var/www/html/asterisk-socket-server/q.wav` против локального сервера с `TONE_DECODE_PROCS=4` и `=0` — сравнить T-one final/partials (должны совпасть при beam_width=200).

---

## Self-Review (заполнено автором плана)

**Покрытие спека:**
- `TONE_DECODE_PROCS`/`TONE_BEAM_WIDTH` → Task 1 ✓
- резолвер kenlm + фабрика пула (gate) + worker → Task 2 ✓
- forward_split (стейдж A, без декода) + decode_async (стейдж B) → Task 3 ✓
- inferer через пул + фолбэк + финализация → Task 4 ✓
- пул в lifespan (create/close) → Task 5 ✓
- kenlm mmap / бэклог / идентичность текста → Task 6 (живая) ✓
- гейт `TONE_DECODE_PROCS=0` = текущий путь → Task 2 (None) + Task 4 (фолбэк) ✓

**Плейсхолдеры:** нет (код приведён целиком).

**Согласованность имён:** `_kenlm_path`, `make_decode_pool`/`get_decode_pool`, `_decode_pool_init`,
`_decode_worker`, `forward_split`/`forward_split_async`, `decode_async`, `_Phrase`,
`app.state.tone_decode_pool`, `TONE_DECODE_PROCS`/`TONE_BEAM_WIDTH` — определены в ранних задачах
и используются согласованно в поздних. ✓
