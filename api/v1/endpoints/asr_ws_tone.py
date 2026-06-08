# -*- coding: utf-8 -*-
"""
WebSocket-роут /api/v1/asr/ws-stream — НАСТОЯЩИЙ потоковый риалтайм на нативном T-one.

В отличие от /api/v1/asr/ws (офлайн псевдо-стрим GigaAM: копит до MAX_OVERLAP_DURATION и
распознаёт целым куском), здесь аудио скармливается модели кадрами по 300 мс с сохранением
состояния, а фразы отдаются по мере их завершения детектором границ T-one (~0.3-1 c).

Особенности:
  - Автодетект протокола (services/ws_protocol): понимает и legacy ({config}/{eof}/raw bytes),
    и новый ({type:config/audio_chunk/eos/ping}). Ответы в формате WSResultMessage —
    его поля (silence/data/last_message) совместимы с legacy asterisk-socket-server.
  - Потоковый ресемплинг источник->8 кГц (T-one фиксирован на 8 кГц).
  - Времена фраз накопительны от первого пакета (по объёму поданного аудио) — корректны
    для поканального мёржа на стороне клиента.
  - БД не используется (в отличие от asr_ws.py) — эндпоинт независим от наличия таблиц.
"""

import asyncio
import logging
import uuid

from fastapi import APIRouter, WebSocket

from config import settings
from models.ws_models import (
    WSResultMessage,
    WSRecognitionData,
    WSPongMessage,
    WSMessageType,
)
from services.ws_manager import ConnectionManager
from services.ws_protocol import detect
from utils.tone_stream import take_frames, flush_tail, phrase_to_data, StreamResampler, forward_async, finalize_async
from Recognizer.tone_engine import get_tone_pipeline

router = APIRouter(prefix="/asr", tags=["ASR"])
logger = logging.getLogger(__name__)


def _result_message(phrase, channel_name: str, last: bool = False) -> WSResultMessage:
    return WSResultMessage(
        type=WSMessageType.final_result if last else WSMessageType.partial_result,
        channel_name=channel_name,
        silence=False,
        data=WSRecognitionData(**phrase_to_data(phrase)),
        last_message=last,
    )


@router.websocket("/ws-stream")
async def websocket_tone_stream(websocket: WebSocket):
    """
    Приём потока развязан с инференсом: задача reader всегда дренирует сокет
    (контрол — config/ping/eos — обрабатывается сразу, аудио-кадры кладутся в
    очередь без блокировки), задача inferer отдельно гонит T-one и шлёт фразы.
    Так пока коннект ждёт инференс, его сокет продолжает читаться -> pong'и
    вычитываются -> uvicorn не рвёт коннект по keepalive (1011) под нагрузкой.
    """
    manager: ConnectionManager = websocket.app.state.ws_manager
    client_id = str(uuid.uuid4())

    if not await manager.connect(websocket, client_id):
        return  # лимит соединений исчерпан, ConnectionManager уже закрыл сокет

    # Предзагруженный в lifespan движок (готов сразу после старта); фолбэк - ленивая загрузка
    pipeline = getattr(websocket.app.state, "tone_pipeline", None) or get_tone_pipeline()
    executor = websocket.app.state.tone_executor

    audio_q: asyncio.Queue = asyncio.Queue()  # элементы (samples, is_last); None — сентинел конца
    send_lock = asyncio.Lock()                # сериализация отправки (reader-pong vs inferer-фразы)
    ctx = {"channel": "Null"}                 # channel_name, общий reader -> inferer

    async def send(message) -> None:
        # Отправка атомарна: Starlette не любит конкурентный send из двух задач.
        async with send_lock:
            await manager.send_message(client_id, message)

    async def reader() -> None:
        """Дренирует сокет: контрол сразу, аудио-кадры -> очередь (без блокировки на инференсе)."""
        buf = bytearray()
        resampler = StreamResampler(settings.TONE_SAMPLE_RATE)  # проходной, пока не пришёл config
        try:
            while True:
                try:
                    message = await websocket.receive()
                except Exception as exc:
                    logger.debug("[tone] receive error %s: %s", client_id, exc)
                    break

                evt = detect(message)

                if evt.kind == "disconnect":
                    logger.info("[tone] disconnect %s (%s)", ctx["channel"], client_id)
                    break

                if evt.kind == "config":
                    ctx["channel"] = evt.channel_name or "Null"
                    sr = evt.sample_rate or settings.TONE_SAMPLE_RATE
                    resampler = StreamResampler(sr)
                    if sr != settings.TONE_SAMPLE_RATE:
                        logger.info("[tone] %s: sample_rate=%s, включён ресемплинг к 8 кГц", ctx["channel"], sr)
                    logger.info("[tone] config received for channel %s", ctx["channel"])
                    continue

                if evt.kind == "ping":
                    await send(WSPongMessage())  # сразу, не за инференсом -> keepalive жив
                    continue

                if evt.kind == "audio":
                    # Время T-one считается по объёму поданного аудио (от первого пакета),
                    # ресемплинг сохраняет длительность — метки остаются корректными.
                    buf.extend(resampler.process(evt.audio or b""))
                    for samples in take_frames(buf):
                        audio_q.put_nowait((samples, False))  # без блокировки: сокет дренируется дальше
                    continue

                if evt.kind == "eos":
                    logger.info("[tone] EOS for channel %s", ctx["channel"])
                    break

                # evt.kind == "ignore" — молча пропускаем
        finally:
            # Дослать хвост ресемплера/буфера как is_last-кадр, затем сентинел конца.
            try:
                buf.extend(resampler.process(b"", last=True))
                tail = flush_tail(buf)
                if tail is not None:
                    audio_q.put_nowait((tail, True))
            except Exception as exc:
                logger.error("[tone] flush tail error %s (%s): %s", ctx["channel"], client_id, exc)
            audio_q.put_nowait(None)  # сентинел: inferer финализирует и завершится

    async def inferer() -> None:
        """Потребляет очередь: forward -> партиалы; на сентинеле — finalize + финальный last_message."""
        state = None
        final_phrases = []
        while True:
            item = await audio_q.get()
            if item is None:
                break
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

        # --- финализация ---
        try:
            fin_phrases, state = await finalize_async(executor, pipeline, state)
            final_phrases.extend(fin_phrases)
        except Exception as exc:
            logger.error("[tone] finalize error %s (%s): %s", ctx["channel"], client_id, exc)

        final_phrases = [p for p in final_phrases if p.text]
        for p in final_phrases[:-1]:
            await send(_result_message(p, ctx["channel"]))
        if final_phrases:
            await send(_result_message(final_phrases[-1], ctx["channel"], last=True))
        else:
            await send(WSResultMessage(
                type=WSMessageType.final_result,
                channel_name=ctx["channel"],
                silence=True,
                data=WSRecognitionData(),
                last_message=True,
            ))

    logger.debug("[tone] new stream %s", client_id)
    try:
        await asyncio.gather(reader(), inferer())
    finally:
        await manager.disconnect(client_id)
        logger.info("[tone] closed %s (%s)", ctx["channel"], client_id)
