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
