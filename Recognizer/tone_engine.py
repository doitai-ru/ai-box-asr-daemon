# -*- coding: utf-8 -*-
"""
Ленивый синглтон для потокового движка T-one (tone.StreamingCTCPipeline).

Грузится один раз при первом обращении (а не при импорте), чтобы:
  - не тянуть модель, когда USE_TONE_STREAMING выключен;
  - не вмешиваться в загрузку основной (офлайн) ASR-модели GigaAM в Recognizer/__init__.py.

T-one работает на CPU (так задумано библиотекой; для чанков по 300 мс это оптимально).
Модель и KenLM качаются с HF (t-tech/T-one) в каталог HF_HOME (= ./models, см. config.py).
"""

import config
from utils.do_logging import logger

_pipeline = None


def get_tone_pipeline():
    """Возвращает singleton StreamingCTCPipeline, загружая его при первом вызове."""
    global _pipeline
    if _pipeline is None:
        from tone import StreamingCTCPipeline, DecoderType

        decoder_type = (DecoderType.GREEDY
                        if str(config.TONE_DECODER).lower() == "greedy"
                        else DecoderType.BEAM_SEARCH)
        logger.info(f"Загрузка потоковой модели T-one (decoder={decoder_type.value})...")
        _pipeline = StreamingCTCPipeline.from_hugging_face(decoder_type=decoder_type)
        logger.info("Потоковая модель T-one загружена.")
    return _pipeline
