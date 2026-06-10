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

# Арена-опции CUDA, чтобы видеопамять не росла неограниченно (kSameAsRequested = без over-allocation)
CUDA_PROVIDERS = [
    ("CUDAExecutionProvider", {
        "arena_extend_strategy": "kSameAsRequested",
        "do_copy_in_default_stream": True,
    }),
    "CPUExecutionProvider",
]


def _model_dir() -> str:
    return os.path.join(settings.HF_HOME, "tone")


def _repo_dir() -> str:
    """Корень репозитория (tone_engine.py лежит в {repo}/Recognizer/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fp32_model_path() -> str:
    """Путь до запечённой в git fp32-модели T-one (см. tools/convert_tone_fp32.py)."""
    return os.path.join(_repo_dir(), "models", "tone_fp32", "model.onnx")


def _build_pipeline():
    import onnxruntime as ort
    from tone import StreamingCTCPipeline, DecoderType
    from tone.onnx_wrapper import StreamingCTCModel
    from tone.logprob_splitter import StreamingLogprobSplitter
    from tone.decoder import GreedyCTCDecoder, BeamSearchCTCDecoder

    greedy = str(settings.TONE_DECODER).lower() == "greedy"
    decoder_type = DecoderType.GREEDY if greedy else DecoderType.BEAM_SEARCH

    # fp32-модель (TONE_FP32): убирает 38 Memcpy-нод на CUDA (fp16<->fp32 Cast'ы на CPU).
    # Состояние модели тоже fp32 -> штатный StreamingCTCModel хардкодит fp16, поэтому сабкласс.
    fp32_path = _fp32_model_path()
    if getattr(settings, "TONE_FP32", False) and os.path.exists(fp32_path):
        if settings.STREAM_WITH_GPU:
            sess = ort.InferenceSession(fp32_path, providers=CUDA_PROVIDERS)
        else:
            sess = ort.InferenceSession(fp32_path)
        logger.info("T-one fp32-модель: %s (провайдер %s)", fp32_path, sess.get_providers()[0])

        class _FP32StreamingModel(StreamingCTCModel):
            """state в fp32 (родитель хардкодит fp16): init нулями fp32, без fp16-проверки."""
            def forward(self, audio_chunk, state):
                if state is None:
                    import numpy as _np
                    state = _np.zeros((audio_chunk.shape[0], self.STATE_SIZE), dtype=_np.float32)
                return self._ort_sess.run(None, {"signal": audio_chunk, "state": state})

        model = _FP32StreamingModel(sess)
        splitter = StreamingLogprobSplitter()
        decoder = GreedyCTCDecoder() if greedy else BeamSearchCTCDecoder.from_local(_kenlm_path())
        return StreamingCTCPipeline(model, splitter, decoder)

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


# --- decode-пул: kenlm beam-search в отдельных процессах (раздельные GIL) ---

import glob
from concurrent.futures import ProcessPoolExecutor

_decode_pool = None
_worker_decoder = None  # в каждом процессе пула


def _kenlm_path() -> str | None:
    """Путь до kenlm.bin: {HF_HOME}/tone/kenlm.bin, иначе из HF-кэша hub."""
    direct = os.path.join(_model_dir(), "kenlm.bin")
    if os.path.exists(direct):
        return direct
    hits = glob.glob(os.path.join(settings.HF_HOME, "hub", "models--t-tech--T-one",
                                  "snapshots", "*", "kenlm.bin"))
    return hits[0] if hits else None


def _decode_pool_init(kenlm_path: str) -> None:
    """initializer воркера: сброс унаследованных сигналов + загрузка декодера.

    Воркеры форкаются ПОСЛЕ того, как uvicorn поставил обработчик SIGTERM, и наследуют
    его -> на рестарте игнорируют SIGTERM systemd и висят до SIGKILL (90с). Возвращаем
    дефолтную диспозицию: SIGTERM -> терминация, SIGINT воркер игнорирует (Ctrl-C -> main).
    """
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global _worker_decoder
    from tone.decoder import BeamSearchCTCDecoder
    _worker_decoder = BeamSearchCTCDecoder.from_local(kenlm_path)


def _decode_worker(logprobs, beam_width: int) -> str:
    """Декод одной фразы в процессе пула (configurable beam_width поверх хардкода tone)."""
    return _worker_decoder._decoder.decode(logprobs, beam_width=beam_width)


def make_decode_pool():
    """ProcessPoolExecutor под декод (или None при TONE_DECODE_PROCS<=0 / отсутствии kenlm)."""
    procs = int(getattr(settings, "TONE_DECODE_PROCS", 0) or 0)
    if procs <= 0:
        return None
    kpath = _kenlm_path()
    if not kpath:
        logger.warning("decode-пул выключен: kenlm.bin не найден")
        return None
    logger.info("T-one decode-пул: %s процессов (kenlm=%s)", procs, kpath)
    return ProcessPoolExecutor(max_workers=procs, initializer=_decode_pool_init, initargs=(kpath,))


def get_decode_pool():
    """Ленивый синглтон decode-пула."""
    global _decode_pool
    if _decode_pool is None:
        _decode_pool = make_decode_pool()
    return _decode_pool
