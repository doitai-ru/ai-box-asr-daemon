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
