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
