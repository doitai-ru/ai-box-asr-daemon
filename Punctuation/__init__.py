from utils.pre_start_init import paths
from utils.do_logging import logger
from .punctuate import SbertPuncCaseOnnx
import config

try:
    sbertpunc = SbertPuncCaseOnnx(paths.get("punctuation_model_path"), use_gpu = config.PUNCTUATE_WITH_GPU)
except Exception as e:
    # Не валим импорт всего приложения, если модель пунктуации не скачана/не загрузилась:
    # пунктуация - опциональный шаг офлайн-пути, потоковый T-one от неё не зависит.
    sbertpunc = None
    logger.error(f"Error getting punctuation model - {e}. Пунктуация будет отключена (sbertpunc=None).")
else:
    logger.info(f'Успешно загружена модель Пунктуации')