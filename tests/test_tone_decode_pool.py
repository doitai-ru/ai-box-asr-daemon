# -*- coding: utf-8 -*-
"""Тесты T-one decode-пула."""

import asyncio
import concurrent.futures as cf

import numpy as np
import pytest


def test_config_decode_pool_defaults():
    from config import Settings
    s = Settings()
    assert s.TONE_DECODE_PROCS == 6      # дефолт: пул из 6 процессов (~10 соединений)
    assert s.TONE_BEAM_WIDTH == 200      # как в библиотеке tone


def test_config_decode_pool_overridable():
    from config import Settings
    s = Settings(TONE_DECODE_PROCS=4, TONE_BEAM_WIDTH=50)
    assert s.TONE_DECODE_PROCS == 4
    assert s.TONE_BEAM_WIDTH == 50


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
