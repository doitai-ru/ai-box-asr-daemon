# Вынос инференса T-one из event-loop (offload в выделенный поток)

- **Дата:** 2026-06-08
- **Статус:** дизайн утверждён, готов к плану реализации
- **Ветка:** feature/prod-deploy

## Проблема

Потоковый эндпоинт `/api/v1/asr/ws-stream` (нативный T-one) выполняет инференс
**прямо в event-loop'е** — `pipeline.forward(...)` вызывается синхронно внутри async-обработчика
(`api/v1/endpoints/asr_ws_tone.py:100,121,123`). Прод запускает **один** uvicorn-процесс
(`Docker/Dockerfile:197` → `python3 main.py` → `uvicorn.run(...)` в `main.py:270`), то есть
один event-loop обслуживает всё.

Под нагрузкой синхронный GPU/CPU-инференс держит loop-поток → uvicorn/websockets опаздывает
с протокольным ping и поздно читает pong → его keepalive-таска по истёкшим дедлайнам **разом
рвёт все сокеты с кодом 1011** ("keepalive ping timeout"), включая здоровую офлайн-ручку `/ws`.
Аудио в зачёт keepalive не идёт, поэтому активный поток данных от смерти не спасает.

Корень — **голодание event-loop'а**. Для сравнения, GigaAM-путь (`/ws`, `/api/v1/asr/ws`)
уже уводит инференс в поток через `asyncio.to_thread` / `run_in_executor`
(`Recognizer/engine/stream_recognition.py:57-59`, `Punctuation/__init__.py:144-150`) — у него
этой проблемы нет.

## Цель шага

Снять инференс T-one с event-loop в выделенный поток-исполнитель, чтобы loop всегда успевал
обслуживать ping/pong. Beam-search декодер (KenLM) остаётся декодером по умолчанию.

## Вне области этого шага (отдельные задачи)

- **Multi-GPU / распределение по картам** — следующий этап (процесс на карту = честный
  параллелизм beam-search через раздельные GIL).
- **nginx / реверс-прокси перед приложением.**
- **Тюнинг uvicorn ws-ping** (ws_ping_interval/ws_ping_timeout).

Этот шаг — фундамент под multi-GPU: каждый будущий процесс-на-карту получит свой собственный
`tone_executor` и свой пайплайн на своей карте.

## Установленный факт: пайплайн T-one потокобезопасен для общего синглтона

Анализ исходников `tone` / `pyctcdecode` / `kenlm` (версии в `venv/`):

1. **`StreamingCTCModel.forward`** (`tone/onnx_wrapper.py`) — только `self._ort_sess.run(...)`.
   `onnxruntime.InferenceSession.run` потокобезопасен; собственного изменяемого состояния нет.
2. **`StreamingLogprobSplitter.forward`** (`tone/logprob_splitter.py`) — всё состояние
   во входном/возвращаемом `state`; на `self` только неизменяемые константы.
3. **`BeamSearchCTCDecoder` → `pyctcdecode`**: кэши `cached_lm_scores`/`cached_p_lm_scores`/`beams`
   — **локальные на каждый вызов** (`pyctcdecode/decoder.py:401-412`), не на `self`. Из `self`
   читается только неизменяемое (`_idx2vocab`, LM через read-only `model_container`).
4. **`LanguageModel.score` / kenlm** (`pyctcdecode/language_model.py:308-326`): на каждый вызов
   создаётся свежий `end_state`; `kenlm.BaseScore` — `const`, модель после загрузки неизменяема.
   Мутацию `alpha/beta` делает только `reset_params`, который в рантайме не вызывается.
5. **В стриминге декодер вызывается пофразно без `lm_start_state`** (`tone/pipeline.py`:
   `self.decoder.forward(phrase.logprobs)`) — каждая фраза декодится с нуля, самодостаточно.

**Вывод:** общий синглтон `StreamingCTCPipeline` можно безопасно вызывать конкурентно из
нескольких потоков при любом декодере (greedy и beam_search). Per-connection пайплайны/декодеры
не нужны. Гонок нет.

### Нюанс производительности (не корректности)

Питоновский цикл beam-search держит GIL: ONNX-часть GIL отпускает и перекрывается между
потоками, а сам перебор лучей — нет. Поэтому реальный параллелизм beam-search внутри одного
процесса ограничен GIL; честное масштабирование даёт только мультипроцесс (будущий multi-GPU).
На корректность offload'а это не влияет.

## Дизайн

**Выделенный `ThreadPoolExecutor` под инференс T-one, по умолчанию 1 воркер (настраивается).**

Почему 1 воркер по умолчанию (хотя гонок нет и можно больше): один воркер полностью снимает
инференс с loop-потока и **минимально конкурирует с loop'ом за GIL** → ping/pong отвечаются
вовремя, инцидент 1011 лечится. Пропускная способность beam-декода в одном процессе всё равно
GIL-bound, поэтому больше воркеров её почти не поднимут, зато отъедают у loop'а долю GIL.
Инсталляции на `greedy` смогут поднять число воркеров без правок кода.

### Изменения

1. **`config.py`** — новый параметр:
   ```python
   # Число потоков-воркеров под инференс T-one (вынос с event-loop).
   # 1 — безопасный дефолт (минимум конкуренции за GIL с event-loop'ом).
   # При greedy-декодере можно поднять. Параллелизм beam-search всё равно ограничен GIL.
   TONE_INFER_WORKERS: int = 1
   ```

2. **`main.py` (lifespan)** — рядом с инициализацией/прогревом T-one:
   - создать `app.state.tone_executor = ThreadPoolExecutor(max_workers=settings.TONE_INFER_WORKERS, thread_name_prefix="tone")`;
   - на shutdown — `app.state.tone_executor.shutdown(wait=True)` (после `disconnect_all`).

3. **`api/v1/endpoints/asr_ws_tone.py`** — вынос вызовов инференса в исполнитель:
   - получить `loop = asyncio.get_running_loop()` и `executor = websocket.app.state.tone_executor`;
   - аудио-цикл: материализовать кадры (`frames = list(take_frames(buf))`), затем по очереди
     `phrases, state = await loop.run_in_executor(executor, pipeline.forward, samples, state)`
     (порядок `state` на коннект сохраняется — await'им последовательно);
   - финализация: `pipeline.forward(tail, state, is_last=True)` и `pipeline.finalize(state)`
     обернуть в `run_in_executor` (для `is_last=True` использовать `functools.partial`);
   - отправка результатов (`manager.send_message`) остаётся на loop'е (она и так async).

   Хрупкая логика прокидывания `state` и формирования фраз не меняется — только точка вызова.

## Критерии приёмки

- Под нагрузкой на `/api/v1/asr/ws-stream` event-loop не голодает: офлайн-ручка `/ws` и другие
  потоки больше не падают каскадом с 1011.
- Результаты распознавания (партиалы/финал, тайминги, текст) идентичны прежним при том же входе
  (beam-search декодер по умолчанию).
- `TONE_INFER_WORKERS=1` по умолчанию; значение читается из окружения/.env.
- Исполнитель корректно создаётся на старте и закрывается на shutdown (нет утечки потоков).

## Риски и заметки

- При `TONE_INFER_WORKERS>1` — корректно (пайплайн потокобезопасен), но рост числа воркеров
  усиливает конкуренцию за GIL с event-loop'ом; держим дефолт 1.
- Не использовать общий пул `asyncio.to_thread` для T-one: он делит воркеры с GigaAM-offload'ом
  и не контролируется по размеру — берём отдельный executor.
- `WS_PING_TIMEOUT_SEC` (app-level, `ConnectionManager`) — это НЕ протокольный ping uvicorn;
  на корень инцидента он не влияет и в этом шаге не трогается.
