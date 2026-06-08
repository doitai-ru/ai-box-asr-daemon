# -*- coding: utf-8 -*-
"""
Бакет-паддинг семплов перед ASR (GigaAM).

Зачем: вход ONNX-модели имеет динамическую ось времени (seq_len). На каждую новую
длину onnxruntime/cuDNN заново ищет алгоритмы свёрток (EXHAUSTIVE) и выделяет арену
под форму — отсюда спайки латентности и набор видеопамяти под разные размеры. Если
кормить модель только фиксированным набором длин (бакетов) и на тех же бакетах её
прогреть, в рантайме холодного поиска нет, память стабильна.

Нарезку входа до <= макс. бакета по границам речи делает апстрим-VAD
(utils.chunk_doing.find_last_speech_position). Здесь — только паддинг чанка нулями
в хвост до ближайшего сверху бакета. Реальные слова лежат в ранних кадрах, хвост
даёт CTC-бланки, поэтому таймкоды реальных слов не сдвигаются.

Чанк длиннее макс. бакета апстрим-VAD не порождает; на всякий случай такой возвращается
как есть — без обрезки (не теряем аудио), ценой редкой нефиксированной формы.
"""

import numpy as np


def pad_to_bucket(samples: np.ndarray, sample_rate: int, buckets) -> np.ndarray:
    """
    Паддит samples нулями в хвост до ближайшего сверху бакета.

    :param samples: 1-D массив семплов (обычно float32)
    :param sample_rate: частота дискретизации (Гц)
    :param buckets: список длин бакетов в секундах, напр. [4, 8, 16, 30]; пустой -> выкл.
    :return: паддённый массив (или исходный, если паддинг не нужен/выключен)
    """
    if not buckets or len(samples) == 0:
        return samples

    bucket_samples = sorted(int(b * sample_rate) for b in buckets)
    target = next((b for b in bucket_samples if b >= len(samples)), None)

    if target is None or target == len(samples):
        # длиннее макс. бакета (без обрезки) либо ровно бакет — отдаём как есть
        return samples

    padded = np.zeros(target, dtype=samples.dtype)
    padded[:len(samples)] = samples
    return padded
