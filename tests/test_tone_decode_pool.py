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
