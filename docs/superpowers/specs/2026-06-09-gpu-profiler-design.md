# GPU-профилировщик: поканальная инструментация + мониторинг боя

- **Дата:** 2026-06-09
- **Статус:** дизайн утверждён, готов к плану реализации
- **Ветка:** feature/gpu-profiler

## Проблема

На проде видеопамять растёт под нагрузкой, и нужно на бою видеть **на чём она растёт**:
сколько держит сам процесс, как это коррелирует с конкуренцией (коннекты по путям),
с бэклогом инференса и с тем, какой компонент в этот момент работает (VAD, диаризация,
пунктуация, GigaAM `/ws`, T-one `/ws-stream`).

Замеры показали: память = функция **конкуренции** (число одновременных инференсов),
плато, а не утечка; arena onnxruntime не отдаёт побайтовую разбивку. Поэтому атрибуция
строится **по корреляции во времени**: пишем текущую память процесса + что делаем, и по
таймлайну видно, кто двигает память. Дельта до/после вызова не нужна (под конкуренцией
шумит) и не считается.

## Цель

Лёгкая, выключаемая инструментация, которая на бою даёт временной ряд:
`{память процесса, used/free карты, util, коннекты по путям, бэклог T-one, активные
компоненты}` + строки компонентов «что делаю + текущая память». Плюс эндпоинт-снимок и
standalone-монитор для съёма кривой и роста/плато.

## Ограничения и принципы

- **Гейт `GPU_PROFILE` (env, по умолчанию выкл):** при выкл — нулевые накладные (`record`
  это no-op, фон-сэмплер не стартует). Включаем только на расследование.
- **NVML не на горячем пути:** «текущую память» компонентные логи берут из **кэш-снимка**,
  обновляемого фоновым сэмплером раз в ~1с. На вызов компонента — только чтение кэша.
- **Строим на существующем:** `pynvml~=12` уже в зависимостях; `SystemMetricsCollector`,
  `ConnectionManager`, `metrics_reporter` уже есть — переиспользуем хуки, не дублируем.
- Изоляция: профилировщик — отдельный модуль с узким интерфейсом (`profile()`, `record()`,
  `gpu_sampler_loop`, `snapshot()`); компоненты зависят только от `profile()`.

## Архитектура

### 1. Модуль `core/gpu_profiler.py`

Состояние модуля (in-memory):
- `_snapshot: dict` — последний снимок: `process_gpu_mib`, `gpu_used_mib`, `gpu_free_mib`,
  `gpu_total_mib`, `gpu_util_pct`, `updated_at`.
- `_components: dict[str, dict]` — агрегаты на компонент: `active` (сейчас в работе),
  `calls` (всего), `last_action`, `last_ms`.
- `_enabled: bool` — из `settings.GPU_PROFILE`.

API:
- `enabled() -> bool` — гейт.
- `snapshot() -> dict` — последний кэш-снимок (для эндпоинта).
- `components() -> dict` — копия агрегатов (для эндпоинта).
- `record(component: str, action: str, **fields)` — если выкл, мгновенный return; иначе
  пишет JSONL-строку `{ts, component, action, gpu_mib: _snapshot.process_gpu_mib, **fields}`
  в `logs/gpu_profile.jsonl` (через выделенный logger) и двигает агрегаты.
- `profile(component, **fields)` — async-контекст-менеджер: `record(component,"start",**fields)`
  на входе, `record(component,"end",dur_ms=...)` на выходе; инкремент/декремент `active`.
  При выкл — пустой контекст (без накладных).
- `gpu_sampler_loop(app_state, interval_sec≈1.0)` — фон-таска: читает NVML (process GPU MiB
  по своему PID через `nvmlDeviceGetComputeRunningProcesses`; used/free/total/util карты),
  читает коннекты по путям и бэклог T-one из `app_state`, обновляет `_snapshot`, пишет
  строку `{ts, kind:"sample", ...snapshot, conns_by_path, tone_backlog}` в тот же JSONL.
- `init_nvml()` — ленивый кэш NVML-хендла (как в `is_alive.py`), с фолбэком при отсутствии GPU.

Выделенный logger `gpu_profile` → `logs/gpu_profile.jsonl` (RotatingFileHandler), не засоряет
основной лог.

### 2. Инструментация компонентов (`async with profile(...)` / `with profile(...)`)

Точки (по одному узкому месту на компонент):
- **VAD:** `VoiceActivityDetector/do_vad.py` (инференс silero) → `profile("vad", n=len(samples))`.
- **Диаризация:** `Diarisation/do_diarize.py` (извлечение эмбеддингов) → `profile("diar", batch=...)`.
- **Пунктуация:** `Punctuation/__init__.py::punctuate` → `profile("punct", n=...)`.
- **GigaAM:** `Recognizer/engine/stream_recognition.py::_simple_recognise_sync` →
  `profile("giga", samples=len, bucket=...)`.
- **T-one:** `utils/tone_stream.py::forward_async`/`finalize_async` →
  `profile("tone", n=..., is_last=...)`.

Sync-компоненты (VAD/диар/пункт/GigaAM выполняются в потоках) используют sync-вариант
контекста; T-one async-хелперы — async-вариант. Контекст пишет start/end и держит `active`.

### 3. Корреляционные хуки

- **Коннекты по путям:** `services/ws_manager.py::ConnectionManager.connect(ws, client_id, kind="generic")`
  — новый параметр `kind` (`"giga"`/`"tone"`/`"admin"`), сохраняется в `ConnectionMeta`;
  свойство `counts_by_kind() -> dict`. Вызовы `manager.connect(...)` в обработчиках получают
  свой `kind`. Дефолт `"generic"` сохраняет обратную совместимость.
- **Бэклог T-one:** реестр активных очередей в `core/gpu_profiler.py`:
  `register_queue(q)` / `unregister_queue(q)`; `tone_backlog() = sum(q.qsize())`.
  Обработчик `/ws-stream` регистрирует свой `audio_q` на входе и снимает в `finally`.

### 4. Эндпоинт `/api/v1/admin/gpu-profile`

GET, auth как у остальных admin-ручек (admin/superadmin). Возвращает:
```json
{
  "snapshot": { "process_gpu_mib": ..., "gpu_used_mib": ..., "gpu_free_mib": ...,
                "gpu_total_mib": ..., "gpu_util_pct": ..., "updated_at": ... },
  "components": { "vad": {"active":..,"calls":..,"last_ms":..}, "giga": {...}, "tone": {...}, ... },
  "conns_by_path": { "giga": N, "tone": M, "admin": K, "total": ... },
  "tone_backlog": B,
  "config": { "provider": ..., "arena_extend_strategy": "kSameAsRequested"|null,
              "tone_infer_workers": ..., "asr_pad_buckets_sec": [...], "gpu_profile": true }
}
```
При `GPU_PROFILE=False` эндпоинт отвечает `{"enabled": false}` (снимка нет).

### 5. Standalone-монитор `tools/gpu_monitor.py`

CLI: опрашивает эндпоинт (URL + токен из argv/env) раз в N сек ИЛИ тейлит
`logs/gpu_profile.jsonl`. Выводит консоль-таблицу + пишет CSV. Считает:
- рост/плато: max, дельты, признак «вышло на плато vs лезет дальше»;
- корреляцию: память process_gpu_mib против `conns_by_path.total`/`tone_backlog`.

## Поток данных

```
компоненты ──profile()──┐
                        ├──> logs/gpu_profile.jsonl ──> tools/gpu_monitor.py (тейл/CSV/кривая)
gpu_sampler_loop ───────┘            (строки component + sample)
        │ (кэш _snapshot, _components, conns_by_path, tone_backlog)
        └──────────────────────────────> GET /api/v1/admin/gpu-profile (снимок) ──> монитор/дашборд
```

## Обработка ошибок

- Нет GPU / NVML недоступен → `init_nvml()` ловит, сэмплер пишет `process_gpu_mib=null`,
  сервис работает (профиль деградирует, не падает).
- Любое исключение в `record`/`profile`/сэмплере — ловится и логируется на WARNING, инференс
  не затрагивается (профиль не должен влиять на обслуживание).
- Запись JSONL — best-effort; ошибка записи не пробрасывается в горячий путь.

## Тестирование

- **Юнит (быстрые, без GPU):**
  - `record()`/`profile()` — no-op при `GPU_PROFILE=False` (ничего не пишут, агрегаты пустые).
  - При `GPU_PROFILE=True` (мок снимка, temp-путь лога): `profile()` пишет start+end, двигает
    `active`/`calls`, считает `dur_ms`.
  - `register_queue`/`tone_backlog` — сумма `qsize()` зарегистрированных очередей.
  - `ConnectionManager.counts_by_kind()` — корректный подсчёт по `kind`.
- **Живой прогон (на GPU):** включить `GPU_PROFILE=1`, прогнать 12-поканальный тест,
  проверить, что `logs/gpu_profile.jsonl` содержит sample-строки с растущей памятью и
  component-строки giga/tone, эндпоинт отдаёт снимок, монитор рисует кривую.

## Критерии приёмки

- `GPU_PROFILE=False` (дефолт): нулевые накладные, эндпоинт `{"enabled": false}`, JSONL не пишется.
- `GPU_PROFILE=True`: сэмплер раз в ~1с пишет снимок (process GPU MiB + конкуренция + бэклог);
  компоненты пишут start/end с текущей памятью из кэша; эндпоинт отдаёт снимок+агрегаты+конфиг;
  монитор снимает кривую и различает плато/рост.
- Профиль не влияет на инференс и не роняет сервис при отсутствии GPU/NVML.

## Вне области

- Побайтовая разбивка arena по компонентам (onnxruntime не отдаёт).
- Persist профиля в БД (пишем в JSONL; история метрик в `SystemLog` остаётся как есть).
- Изменение существующих метрик/`/is_alive`/admin-дашборда (отдельная ручка, не трогаем).
- probe-режим (изолированный замер компонента) — возможное расширение позже, в этот шаг не входит.
