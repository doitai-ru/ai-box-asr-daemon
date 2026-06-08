# -*- coding: utf-8 -*-
"""
Ленивый синглтон потокового движка T-one (tone.StreamingCTCPipeline).

Модель лежит в каталоге {HF_HOME}/tone (model.onnx [+ kenlm.bin]) и ЗАПЕКАЕТСЯ туда на
этапе сборки образа (RUN в Dockerfile) — отдельный от hub каталог, чтобы бинд-маунт hub
её не перекрывал. Здесь питон только ГРУЗИТ модель (через from_local). Если каталог пуст
(напр. dev без запекания) — фолбэк на штатную from_hugging_face (она сама докачает в кэш).

Отдельной переменной под модель нет — база одна, HF_HOME.

Провайдер: CPU по умолчанию; CUDAExecutionProvider при settings.STREAM_WITH_GPU=1.
"""

import logging
import os

from config import settings

logger = logging.getLogger(__name__)

_pipeline = None

CUDA_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def _model_dir() -> str:
    return os.path.join(settings.HF_HOME, "tone")


def _build_pipeline():
    import onnxruntime as ort
    from tone import StreamingCTCPipeline, DecoderType
    from tone.onnx_wrapper import StreamingCTCModel
    from tone.logprob_splitter import StreamingLogprobSplitter
    from tone.decoder import GreedyCTCDecoder, BeamSearchCTCDecoder

    greedy = str(settings.TONE_DECODER).lower() == "greedy"
    decoder_type = DecoderType.GREEDY if greedy else DecoderType.BEAM_SEARCH

    model_dir = _model_dir()
    model_path = os.path.join(model_dir, "model.onnx")

    # Не запечено (dev) — штатная загрузка с авто-докачкой в HF-кэш.
    if not os.path.exists(model_path):
        logger.info("T-one не найден в %s — загрузка через from_hugging_face", model_dir)
        if settings.STREAM_WITH_GPU:
            mp = StreamingCTCModel.download_from_hugging_face()
            sess = ort.InferenceSession(mp, providers=CUDA_PROVIDERS)
            decoder = GreedyCTCDecoder() if greedy else BeamSearchCTCDecoder.from_hugging_face()
            return StreamingCTCPipeline(StreamingCTCModel(sess), StreamingLogprobSplitter(), decoder)
        return StreamingCTCPipeline.from_hugging_face(decoder_type=decoder_type)

    # Запечённая модель — грузим локально, без сети.
    if settings.STREAM_WITH_GPU:
        sess = ort.InferenceSession(model_path, providers=CUDA_PROVIDERS)
    else:
        sess = ort.InferenceSession(model_path)
    logger.info("T-one провайдер: %s (локально из %s)", sess.get_providers()[0], model_dir)

    model = StreamingCTCModel(sess)
    splitter = StreamingLogprobSplitter()
    decoder = GreedyCTCDecoder() if greedy else BeamSearchCTCDecoder.from_local(os.path.join(model_dir, "kenlm.bin"))
    return StreamingCTCPipeline(model, splitter, decoder)


def get_tone_pipeline():
    """Возвращает singleton StreamingCTCPipeline, загружая его при первом вызове."""
    global _pipeline
    if _pipeline is None:
        logger.info("Загрузка потоковой модели T-one (decoder=%s, gpu=%s, dir=%s)...",
                    settings.TONE_DECODER, settings.STREAM_WITH_GPU, _model_dir())
        _pipeline = _build_pipeline()
        logger.info("Потоковая модель T-one загружена.")
    return _pipeline
