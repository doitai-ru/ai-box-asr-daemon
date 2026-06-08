# -*- coding: utf-8 -*-
"""
Бакет-нарезка/паддинг семплов перед ASR (GigaAM).

Зачем: вход ONNX-модели имеет динамическую ось времени (seq_len). На каждую новую
длину onnxruntime/cuDNN заново ищет алгоритмы свёрток (EXHAUSTIVE) и выделяет арену
под форму — отсюда спайки латентности и набор видеопамяти под разные размеры. Если
кормить модель только фиксированным набором длин (бакетов) и на тех же бакетах её
прогреть, в рантайме холодного поиска нет, память стабильна.

Контракт:
  - чанк <= максимального бакета  -> один сегмент, допадденный нулями до ближайшего
    сверху бакета;
  - чанк > максимального бакета    -> режется на куски по максимальному бакету (аудио
    НЕ теряется и НЕ обрезается), последний неполный кусок паддится до своего бакета;
  - пустой список бакетов          -> паддинг отключён (сегмент возвращается как есть).

Паддинг только в хвост нулями: реальные слова лежат в ранних кадрах, хвост даёт
CTC-бланки, поэтому таймкоды реальных слов не сдвигаются.
"""

import numpy as np


def split_to_buckets(samples: np.ndarray, sample_rate: int, buckets) -> list:
    """
    Режет samples на сегменты <= макс. бакета и паддит каждый до его бакета.

    :param samples: 1-D массив семплов (обычно float32)
    :param sample_rate: частота дискретизации (Гц)
    :param buckets: список длин бакетов в секундах, напр. [4, 8, 16, 30]
    :return: список np.ndarray-сегментов (для входа <= макс. бакета — один элемент)
    """
    if not buckets or len(samples) == 0:
        return [samples]

    bucket_samples = sorted(int(b * sample_rate) for b in buckets)
    max_b = bucket_samples[-1]

    segments = []
    for start in range(0, len(samples), max_b):
        piece = samples[start:start + max_b]
        target = next((b for b in bucket_samples if b >= len(piece)), max_b)
        if target == len(piece):
            segments.append(piece)
        else:
            padded = np.zeros(target, dtype=samples.dtype)
            padded[:len(piece)] = piece
            segments.append(padded)
    return segments
