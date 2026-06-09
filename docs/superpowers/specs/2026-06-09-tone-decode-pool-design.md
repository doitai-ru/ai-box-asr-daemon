# T-one decode-пул: параллелизация kenlm-декода по процессам

- **Дата:** 2026-06-09
- **Статус:** дизайн утверждён, готов к плану
- **Ветка:** feature/tone-decode-pool

## Проблема

Под реальным трафиком (профилировщик): при 12-14 одновременных T-one стримах
один beam_search-воркер не успевает → бэклог уходит в разнос (~5600 кадров ≈ 13 мин
лага транскрипции). GPU при этом свободна (плато ~5 ГБ из 8) — упираемся **не в GPU**.

**Подтверждено эмпирически, что декод на CPU:**
- `kenlm.so` не линкуется с CUDA (`ldd`); pyctcdecode beam-search — чистый Python.
- 4 потока на декоде → 0.68x (не ускорение): классическая подпись **GIL-bound CPU**
  (GPU-операция отпускала бы GIL → потоки бы перекрылись).
- При декоде GPU-память ±46 МБ (никаких аллокаций).
- Стоимость: ~726 мс/декод при beam_width=200 (L≈120) — декод доминирует.

Итог: узкое место — **kenlm-декод, заперт одним GIL**. Потоки не помогают → нужны
**процессы** (раздельные GIL → параллель по ядрам CPU).

## Цель

Раскидать kenlm-декод по `ProcessPoolExecutor`, оставив GPU-акустику и нарезку фраз
в основном процессе. Одна GPU-модель (не дублируем видеопамять), kenlm mmap-шарится
между воркерами. Параллелизм — между коннектами. Поведение/протокол клиента не меняется.

Выключаемо: `TONE_DECODE_PROCS=0` (дефолт) — текущий in-process путь (фолбэк).

## Ключевой факт о декоде T-one

Декод в T-one **пофразный и независимый**: `decoder.decode(phrase.logprobs)` каждую
фразу декодит с нуля, без `lm_start_state` (проверено в исходниках). Значит фразы можно
декодить параллельно без состояния. Состояние стрима (`model_state`, `splitter_state`)
живёт только в стейдже A (forward+split).

## Архитектура

### Два стейджа на коннект (в inferer'е /ws-stream)

```
кадр -> [A] forward_split (в tone_executor, в процессе):
            model.forward (GPU, отпускает GIL) -> logprobs
            splitter.forward -> logprob-фразы (logprobs + start/end время) + state
        на каждую фразу -> [B] decode в process-pool:
            decode(logprobs, beam_width) -> текст         (раздельные GIL, параллельно)
        -> TextPhrase(текст, start, end) -> send (порядок в рамках коннекта сохраняем)
```

- **Стейдж A** — дёшево, GIL свободен; 1 поток tone_executor тянет много коннектов.
- **Стейдж B** — тяжёлый kenlm-декод, в пуле процессов; параллелится по ядрам.
- Параллелизм в основном **между коннектами** (фразы от разных сессий молотятся разом);
  внутри коннекта фразы редко перекрываются, порядок сохраняем (await в порядке сабмита).

### Компоненты / файлы

- **`Recognizer/tone_engine.py`**
  - `get_tone_decode_pool()` — ленивый синглтон `ProcessPoolExecutor(max_workers=TONE_DECODE_PROCS)`
    с `initializer`, который в каждом воркере грузит `BeamSearchCTCDecoder.from_local(<kenlm.bin>)`
    один раз (kenlm mmap-шарится между процессами); `None`, если `TONE_DECODE_PROCS<=0`.
  - `_decode_worker(logprobs: np.ndarray, beam_width: int) -> str` — функция-воркер:
    глобальный декодер воркера → `_decoder._decoder.decode(logprobs, beam_width=beam_width)`.
  - Путь до `kenlm.bin`: из каталога модели T-one (`{HF_HOME}/tone/kenlm.bin`) либо HF-кэша
    (`hub/models--t-tech--T-one/.../kenlm.bin`) — резолвер по тем же правилам, что `_build_pipeline`.
  - Закрытие пула в lifespan.

- **`utils/tone_stream.py`**
  - `forward_split_async(executor, pipeline, samples, state, is_last)` — в tone_executor:
    `pipeline.model.forward` + `pipeline.logprob_splitter.forward`, **без декода**; возвращает
    `(phrases, state)`, где `phrases` — список `LogprobPhrase`-подобных с `logprobs`, `start_time`,
    `end_time` (конвертация кадры→время скопирована из `pipeline.forward`: `FRAME_SIZE`,
    `MEAN_TIME_BIAS`, `PADDING`, накопленный сдвиг). Результат идентичен `pipeline.forward` минус текст.
  - `decode_async(pool, logprobs, beam_width)` — `loop.run_in_executor(pool, _decode_worker, logprobs, beam_width)`.
  - Фолбэк: если `pool is None` — текущий путь (`forward_async` через `pipeline.forward`).

- **`api/v1/endpoints/asr_ws_tone.py`** (inferer): при наличии пула — `forward_split_async` →
  на каждую фразу `decode_async` (сабмитим в пул) → `TextPhrase` → send; финал после всех фраз.
  При `TONE_DECODE_PROCS=0` — текущий путь без изменений.

- **`config.py`**
  - `TONE_DECODE_PROCS: int = 0` (0 = in-process, дефолт/фолбэк; N>0 = пул из N процессов).
  - `TONE_BEAM_WIDTH: int = 200` (рычаг стоимости декода; используется в `_decode_worker`).

- **`main.py` (lifespan):** прогрев/создание пула при `TONE_DECODE_PROCS>0` и закрытие на shutdown.

## Поток данных / IPC

В пул летят только **logprobs фразы** (numpy `float32`, ~10–60 кадров × 35 ≈ единицы КБ) —
мелкий IPC, не GPU-тензоры, не состояние стрима. Текст возвращается строкой. kenlm в воркерах
mmap-шарит `.bin` (read-only страницы общие → ~5.5 ГБ один раз, не N×).

## Обработка ошибок / порядок

- Исключение в декоде фразы → фраза логируется и пропускается (как сейчас), inferer продолжает.
- Воркер пула умер → `ProcessPoolExecutor` поднимет замену; если декод фразы не удался — пропуск.
- Порядок в рамках коннекта: await декод-future'ов в порядке сабмита фраз; финальный
  `last_message` — после всех.
- Закрытие пула на shutdown (`pool.shutdown(wait=False)`), tone_executor — как сейчас.

## Тестирование

- **Юнит (быстрые):**
  - `forward_split_async` (мок pipeline: `model.forward`/`logprob_splitter.forward` отдают
    фиктивные logprobs/фразы) — возвращает фразы со временем, **decode не вызывается**;
    `state` прокидывается.
  - `decode_async` в реальном мини-`ProcessPoolExecutor` с мок-`_decode_worker` (без kenlm) —
    параллельная отдача текста; порядок сохраняется на стороне inferer'а.
  - конвертация кадры→время идентична `pipeline.forward` на одинаковом входе.
  - гейт `TONE_DECODE_PROCS=0` → старый путь (`forward_async`), новый код не активируется.
- **PoC/живой:** на свободной GPU поднять с `TONE_DECODE_PROCS=4`, 12 поканальных стримов →
  бэклог не копится (или копится кратно медленнее); сравнить с `=0`. Проверить, что RSS пула
  не растёт как N×5.5 ГБ (kenlm mmap-шарится).

## Критерии приёмки

- `TONE_DECODE_PROCS=0` (дефолт): поведение и протокол неизменны; новый код не активен.
- `TONE_DECODE_PROCS=N>0`: декод идёт в пуле процессов, текст/таймкоды идентичны in-process
  пути на том же входе; бэклог под 12 конкурентными растёт кратно медленнее/не растёт.
- `TONE_BEAM_WIDTH` управляет шириной луча в воркере (и в фолбэк-пути).
- Профиль (бэклог/конкуренция) продолжает писаться; финализация без ошибок.
- kenlm в воркерах mmap-шарится (RSS пула << N×5.5 ГБ).

## Вне области

- Изменение протокола клиента (формат partial/final — без изменений).
- Multi-GPU / процесс-на-карту всего приложения (не нужно — GPU не боттлнек).
- Авто-тюнинг beam_width/числа процессов (ставятся конфигом).
- Замена tone_executor (стейдж A остаётся в текущем 1-воркерном executor'е).
