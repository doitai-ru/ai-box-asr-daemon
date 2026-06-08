# -*- coding: utf-8 -*-
"""Тесты бакет-паддинга семплов для GigaAM.

Фикс набор форм -> onnxruntime/cuDNN греется на K бакетов, а не на каждую длину.
Нарезку до <=30с по границам речи делает апстрим-VAD; здесь только паддинг чанка
(<= макс. бакета) нулями в хвост до ближайшего сверху бакета. Чанк длиннее макс.
бакета (апстрим-VAD это предотвращает) возвращается как есть — без обрезки аудио.
"""

import numpy as np

from utils.audio_buckets import pad_to_bucket

SR = 16000
BUCKETS = [4, 8, 16, 30]


def test_pads_up_to_smallest_covering_bucket():
    n = int(2.5 * SR)  # 2.5с -> 4с
    out = pad_to_bucket(np.ones(n, dtype=np.float32), SR, BUCKETS)
    assert len(out) == 4 * SR
    assert out.dtype == np.float32
    assert np.array_equal(out[:n], np.ones(n, dtype=np.float32))  # реальные семплы сохранены
    assert np.all(out[n:] == 0)                                   # хвост — нули


def test_between_buckets_rounds_up():
    n = int(9 * SR)  # 9с -> 16с
    out = pad_to_bucket(np.ones(n, dtype=np.float32), SR, BUCKETS)
    assert len(out) == 16 * SR


def test_exact_bucket_unchanged():
    n = 8 * SR
    a = np.random.randn(n).astype(np.float32)
    out = pad_to_bucket(a, SR, BUCKETS)
    assert np.array_equal(out, a)


def test_max_bucket_unchanged():
    n = 30 * SR
    a = np.random.randn(n).astype(np.float32)
    out = pad_to_bucket(a, SR, BUCKETS)
    assert np.array_equal(out, a)


def test_longer_than_max_returned_as_is_not_truncated():
    n = int(35 * SR)  # > 30с: апстрим-VAD такого не даёт; на всякий случай — без обрезки
    a = np.arange(n, dtype=np.float32)
    out = pad_to_bucket(a, SR, BUCKETS)
    assert len(out) == n
    assert np.array_equal(out, a)


def test_empty_buckets_disabled_passthrough():
    n = int(3 * SR)
    a = np.ones(n, dtype=np.float32)
    out = pad_to_bucket(a, SR, [])
    assert len(out) == n
    assert np.array_equal(out, a)
