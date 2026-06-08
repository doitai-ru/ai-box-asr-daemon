# -*- coding: utf-8 -*-
"""Тесты бакет-нарезки/паддинга семплов для GigaAM.

Фикс набор форм -> onnxruntime/cuDNN греется на K бакетов, а не на каждую длину.
Чанк длиннее самого большого бакета режется на куски <= макс. бакета (без потери
аудио), каждый кусок паддится нулями в хвост до своего бакета.
"""

import numpy as np

from utils.audio_buckets import split_to_buckets

SR = 16000
BUCKETS = [4, 8, 16, 30]


def test_short_chunk_single_segment_padded_to_smallest_covering_bucket():
    n = int(2.5 * SR)  # 2.5с -> бакет 4с, один сегмент
    segs = split_to_buckets(np.ones(n, dtype=np.float32), SR, BUCKETS)
    assert len(segs) == 1
    out = segs[0]
    assert len(out) == 4 * SR
    assert out.dtype == np.float32
    assert np.array_equal(out[:n], np.ones(n, dtype=np.float32))  # реальные семплы сохранены
    assert np.all(out[n:] == 0)                                   # хвост — нули


def test_between_buckets_rounds_up():
    n = int(9 * SR)  # 9с -> 16с
    segs = split_to_buckets(np.ones(n, dtype=np.float32), SR, BUCKETS)
    assert len(segs) == 1
    assert len(segs[0]) == 16 * SR


def test_exact_bucket_unchanged():
    n = 8 * SR
    a = np.random.randn(n).astype(np.float32)
    segs = split_to_buckets(a, SR, BUCKETS)
    assert len(segs) == 1
    assert np.array_equal(segs[0], a)


def test_max_bucket_unchanged_single_segment():
    n = 30 * SR
    a = np.random.randn(n).astype(np.float32)
    segs = split_to_buckets(a, SR, BUCKETS)
    assert len(segs) == 1
    assert np.array_equal(segs[0], a)


def test_longer_than_max_is_split_not_truncated():
    n = int(35 * SR)  # > 30с -> [30с, 5с->паддинг до 8с]
    a = np.arange(n, dtype=np.float32)  # уникальные значения, чтобы проверить порядок
    segs = split_to_buckets(a, SR, BUCKETS)
    assert len(segs) == 2
    assert len(segs[0]) == 30 * SR          # первый кусок ровно макс. бакет
    assert len(segs[1]) == 8 * SR           # остаток 5с -> бакет 8с
    # аудио не потеряно и в правильном порядке: 30с первого + 5с второго == исходные 35с
    real_tail = 5 * SR
    reconstructed = np.concatenate([segs[0], segs[1][:real_tail]])
    assert np.array_equal(reconstructed, a)
    assert np.all(segs[1][real_tail:] == 0)  # хвост последнего куска — нули


def test_empty_buckets_disabled_passthrough():
    n = int(3 * SR)
    a = np.ones(n, dtype=np.float32)
    segs = split_to_buckets(a, SR, [])
    assert len(segs) == 1
    assert np.array_equal(segs[0], a)
