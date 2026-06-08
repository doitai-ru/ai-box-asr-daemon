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
