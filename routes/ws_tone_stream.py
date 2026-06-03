# -*- coding: utf-8 -*-
"""
Потоковый WebSocket-эндпоинт распознавания на нативном T-one.

В отличие от офлайн-пути (ws_audio_transkrib.py), который копит аудио до
MAX_OVERLAP_DURATION и распознаёт целым куском, здесь аудио скармливается модели
кадрами по 300 мс с сохранением состояния (state), а фразы отдаются клиенту по мере
их завершения детектором границ T-one — это даёт настоящий риалтайм (~0.3-1 c).

Всегда доступен на /ws-stream (параллельно текущему офлайн /ws). Дополнительно может
занять и /ws, если включён config.USE_TONE_STREAMING (тогда T-one заменяет офлайн-путь).

Протокол совместим с asterisk-socket-server:
  вход:  текст {"config":{"sample_rate":8000,...},"channelName":...}, затем сырой PCM16,
         в конце текст {"eof":1};
  выход: send_messages(...) -> {channel_name, silence, data:{result,text}, error,
         last_message, sentenced_data}.
"""

import uuid
import ujson

from fastapi import WebSocket

import config
from utils.pre_start_init import app
from utils.do_logging import logger
from utils.send_messages import send_messages
from utils.tone_stream import take_frames, flush_tail, phrase_to_data, StreamResampler
from Recognizer.tone_engine import get_tone_pipeline


async def tone_stream_ws(ws: WebSocket):
    client_id = uuid.uuid4()
    pipeline = get_tone_pipeline()

    state = None
    buf = bytearray()
    channel_name = "Null"
    sample_rate = config.TONE_SAMPLE_RATE
    resampler = StreamResampler(sample_rate)  # проходной, пока не пришёл config

    await ws.accept()
    logger.debug(f"[tone] Принят новый сокет id = {client_id}")

    while True:
        try:
            message = await ws.receive()
        except Exception as wse:
            logger.error(f"[tone] receive error - {wse}")
            return

        if message.get("type") == "websocket.disconnect":
            logger.info(f"[tone] Канал {channel_name} закрыт извне")
            break

        text = message.get("text")
        data_bytes = message.get("bytes")

        # --- управляющие текстовые сообщения ---
        if text:
            try:
                if "config" in text:
                    parsed = ujson.loads(text)
                    cfg = parsed.get("config", {})
                    sample_rate = cfg.get("sample_rate", sample_rate)
                    channel_name = parsed.get("channelName", "Null")
                    resampler = StreamResampler(sample_rate)  # источник->8 кГц при необходимости
                    if sample_rate != config.TONE_SAMPLE_RATE:
                        logger.info(
                            f"[tone] sample_rate={sample_rate} != {config.TONE_SAMPLE_RATE}; "
                            f"включён потоковый ресемплинг к 8 кГц")
                    logger.info(f"[tone] Task received, config - {text}")
                    continue
                elif "eof" in text:
                    logger.info(f"[tone] EOF received in channel {channel_name}")
                    break
                else:
                    logger.error(f"[tone] Не распознан текст сообщения {text} в канале {channel_name}")
                    continue
            except Exception as e:
                logger.error(f"[tone] Ошибка разбора текстового сообщения {text} - {e}")
                continue

        # --- аудио-байты ---
        elif data_bytes:
            buf.extend(resampler.process(data_bytes))  # источник->8 кГц (проходной если уже 8к)
            try:
                # Семантика времени: T-one считает start_time/end_time фраз накопительно
                # по поданному аудио (счётчик кадров в state). Ноль = первый аудиопакет
                # этого канала/сокета, далее время растёт строго по объёму поданного аудио -
                # не зависит от размера пакетов и от wall-clock. Это совместимо с офлайн-путём
                # (time_shift=audio_duration). Межканальный общий ноль собирает клиент
                # (asterisk-socket-server) как audioStreamStartAt[channel] + word.start.
                for samples in take_frames(buf):
                    phrases, state = pipeline.forward(samples, state)
                    for phrase in phrases:
                        if not phrase.text:
                            continue
                        if not await send_messages(ws, _silence=False,
                                                   _data=phrase_to_data(phrase),
                                                   _error=None, _channel_name=channel_name):
                            logger.error("[tone] send_message not ok, work canceled")
                            return
            except Exception as e:
                logger.error(f"[tone] Ошибка распознавания чанка - {e} в канале {channel_name}")
            continue

        else:
            logger.error(f"[tone] Не удалось разобрать сообщение - {message} в канале {channel_name}")

    # --- финализация: дослать хвост и закрыть фразы ---
    last_data = None
    try:
        buf.extend(resampler.process(b"", last=True))  # дослать хвост фильтра ресемплера
        tail = flush_tail(buf)
        final_phrases = []
        if tail is not None:
            phrases, state = pipeline.forward(tail, state, is_last=True)
            final_phrases.extend(phrases)
        fin_phrases, state = pipeline.finalize(state)
        final_phrases.extend(fin_phrases)

        # все фразы, кроме последней, шлём как обычные; последнюю пометим last_message
        for phrase in final_phrases:
            if phrase.text:
                last_data = phrase_to_data(phrase)
    except Exception as e:
        logger.error(f"[tone] Ошибка финализации - {e} в канале {channel_name}")

    is_silence = last_data is None
    try:
        await send_messages(ws, _silence=is_silence, _data=last_data, _error=None,
                            _last_message=True, _channel_name=channel_name)
    except Exception as e:
        logger.error(f"[tone] Ошибка отправки финального сообщения - {e}")

    logger.info(f"[tone] Closing connection {channel_name}")
    try:
        await ws.close()
    except Exception:
        pass
    return


# /ws-stream - потоковая ручка T-one, доступна всегда (параллельно офлайн /ws)
app.websocket("/ws-stream")(tone_stream_ws)

# /ws - занимаем потоковым обработчиком только если включён флаг
if config.USE_TONE_STREAMING:
    app.websocket("/ws")(tone_stream_ws)
    logger.info("[tone] T-one стрим зарегистрирован на /ws (USE_TONE_STREAMING=1)")
