# Базовый образ без CUDA и cuDNN
FROM python:3.12-slim-bullseye

# Источник пакетов для дебиан
RUN sed -i 's/deb.debian.org/mirror.yandex.ru/g' /etc/apt/sources.list

# Установка зависимостей
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    portaudio19-dev \
    python3-pyaudio \
    ffmpeg \
    python3-pip \
    curl \
    ca-certificates \
    jq \
    && rm -rf /var/lib/apt/lists/*  # Очистка кеша apt

# T-one опционален: сборочные зависимости для kenlm ставим только при WITH_TONE=1
ARG WITH_TONE=1
RUN if [ "$WITH_TONE" = "1" ] || [ "$WITH_TONE" = "true" ]; then \
        apt-get update && \
        apt-get install -y build-essential cmake libboost-all-dev libeigen3-dev && \
        rm -rf /var/lib/apt/lists/* ; \
    fi

RUN git lfs install

# Собираем модели для пунктуации
RUN mkdir "temp_models" && \
    cd temp_models && \
    git clone https://huggingface.co/Alexanrd/sbert_punc_case_ru_onnx && \
    cd ..

#RUN mkdir "temp_models"
#COPY models/sbert_punc_case_ru_onnx temp_models/sbert_punc_case_ru_onnx

ARG DIAR_MODEL_NAME=voxblink2_samresnet100_ft

ENV DIAR_MODEL_NAME=$DIAR_MODEL_NAME \
    DIAR_API_URL=https://modelscope.cn/api/v1/datasets/wenet/wespeaker_pretrained_models/oss/tree

    # Скачивание модели диаризации.
RUN mkdir -p temp_models/DIARISATION_model && \
    echo "Пытаемся скачать модель $DIAR_MODEL_NAME..." && \
    # 1. Получаем JSON с данными файлов \
    curl -s -L -H "User-Agent: Mozilla/5.0" "$DIAR_API_URL" > /tmp/api_response.json && \
    # 2. Ищем нужный файл в JSON \
    file_url=$(jq -r --arg model "$DIAR_MODEL_NAME.onnx" '.Data[] | select(.Key == $model) | .Url' /tmp/api_response.json) && \
    if [ -z "$file_url" ]; then \
        echo "ОШИБКА: Модель $DIAR_MODEL_NAME не найдена. Доступные модели:"; \
        jq -r '.Data[] | select(.Key | endswith(".onnx")) | "\(.Key) - \(.Size / (1024*1024) | floor) MB"' /tmp/api_response.json | sort; \
        exit 1; \
    fi && \
    # 3. Скачиваем файл \
    echo "Найдена модель, скачиваем по URL: $file_url" && \
    curl  -L "$file_url" -o temp_models/DIARISATION_model/$DIAR_MODEL_NAME.onnx  && \
    # Проверяем успешность скачивания \
    if [ ! -f "temp_models/DIARISATION_model/$DIAR_MODEL_NAME.onnx" ]; then \
        echo "ОШИБКА: Не удалось скачать модель"; \
        exit 1; \
    else \
        echo "УСПЕХ: Модель сохранена в DIARISATION_model/$DIAR_MODEL_NAME.onnx"; \
    fi && \
    rm -f /tmp/api_response.json


#RUN mkdir -p temp_models/DIARISATION_model
#COPY models/DIARISATION_model/voxblink2_samresnet100_ft.onnx temp_models/DIARISATION_model/voxblink2_samresnet100_ft.onnx

# Собираем модели для VAD
RUN mkdir -p temp_models/VAD_silero_v5
RUN curl -L https://github.com/snakers4/silero-vad/raw/v5.0/files/silero_vad.onnx \
         -o temp_models/VAD_silero_v5/silero_vad.onnx
RUN test -f temp_models/VAD_silero_v5/silero_vad.onnx || (echo "File download failed!" && exit 1)

# requirements берём из НАШЕГО форка (там добавлены зависимости новой архитектуры:
# sqlalchemy/aiosqlite/greenlet/email-validator). GIT_TOKEN — токен gl.amulex.ru со
# scope read_api+read_repository, передаётся при сборке: --build-arg GIT_TOKEN=...
ARG GIT_TOKEN
ARG GIT_REF=main
RUN echo "Downloading requirements.txt from our fork ... " && \
    mkdir -p /tmp/app_setup && \
    curl -fsSL -H "PRIVATE-TOKEN: ${GIT_TOKEN}" \
      "https://gl.amulex.ru/api/v4/projects/amulex%2Fml-ai%2Fasr/repository/files/requirements.txt/raw?ref=${GIT_REF}" \
      -o /tmp/app_setup/requirements.txt

## Копируем requirements в временную директорию
#COPY requirements.txt /tmp/app_setup/requirements.txt

# По умолчанию устанавливаем CPU провайдер
ARG PROVIDER=CPU

# Добавляем CUDA и TENSORRT -зависимости в основной файл requirements.txt
RUN if [ "$PROVIDER" = "TENSORRT" ]; then \
        echo "Configuring for TENSORRT..."; \
        # Заменяем CPU-версию onnxruntime на TensorRT-совместимую GPU-версию \
        sed -i 's#^onnxruntime\s*$#onnxruntime-gpu[cuda, cudnn]== 1.23.2#' /tmp/app_setup/requirements.txt; \
        # Добавляем cupy и TensorRT-специфичные пакеты \
        echo "cupy-cuda12x==13.5.1" >> /tmp/app_setup/requirements.txt; \
        echo "tensorrt_libs" >> /tmp/app_setup/requirements.txt; \
    elif [ "$PROVIDER" = "CUDA" ]; then \
        echo "Configuring for CUDA..."; \
        # Заменяем CPU-версию onnxruntime на GPU-версию \
        sed -i 's#^onnxruntime\s*$#onnxruntime-gpu[cuda, cudnn]== 1.23.2#' /tmp/app_setup/requirements.txt; \
        # Добавляем только cupy \
        echo "cupy-cuda12x==13.5.1" >> /tmp/app_setup/requirements.txt; \
    else \
        echo "Using default CPU requirements."; \
    fi

#RUN cat  /tmp/app_setup/requirements.txt
#RUN sleep infinity

# Устанавливаем зависимости, используя кэш(для тестирования и экономии на трафике, но увеличивает размер основномго образа docker)
#RUN --mount=type=cache,target=/root/.cache/pip  \
#    pip install -r  /tmp/app_setup/requirements.txt

# Устанавливаем зависимости без кэша
RUN pip  install --no-cache-dir -r /tmp/app_setup/requirements.txt

# T-one (потоковый путь /api/v1/asr/ws-stream) — только при WITH_TONE=1.
# --no-deps, чтобы не затянуть CPU-onnxruntime поверх onnxruntime-gpu; затем зависимости (kenlm соберётся).
RUN if [ "$WITH_TONE" = "1" ] || [ "$WITH_TONE" = "true" ]; then \
        pip install --no-cache-dir --no-deps "git+https://github.com/voicekit-team/T-one.git" && \
        pip install --no-cache-dir pyctcdecode kenlm miniaudio ; \
    fi

# Принудительная переустановка нужной версии cuDNN поверх установленной для карт на pascal (GTX 1080)
# RUN pip install --force-reinstall nvidia-cudnn-cu12==9.6.0.74

# Устанавливаем основное приложени
# Код приложения — из НАШЕГО форка (там T-one). Каталог сохраняем как ASR_FastAPI_WS_RU
# (на него завязаны WORKDIR/PYTHONPATH/VOLUME), .git удаляем, чтобы токен не утёк в образ.
RUN echo " Forcing cache invalidation at $(date)" && \
    git clone --depth 1 --branch "$GIT_REF" \
      "https://oauth2:${GIT_TOKEN}@gl.amulex.ru/amulex/ml-ai/asr.git" ASR_FastAPI_WS_RU && \
    rm -rf ASR_FastAPI_WS_RU/.git

# COPY . /ASR_FastAPI_WS_RU/

# Переносим модели
RUN mkdir -p ASR_FastAPI_WS_RU/models && \
    cp -r temp_models/* ASR_FastAPI_WS_RU/models/ && \
    rm -rf temp_models && \
    rm -rf /tmp/app_setup



# Установка переменных окружения и даём возможность менять модель диаризации, ASR на этапе сборки.
ARG DIAR_MODEL_NAME=voxblink2_samresnet100_ft

ENV BASE_SAMPLE_RATE=16000 \
    CAN_DIAR=1 \
    DIAR_WITH_GPU=1 \
    IS_PROD=1 \
    LOGGING_LEVEL=INFO \
    MAX_OVERLAP_DURATION=25 \
    NUM_THREADS=0 \
    VAD_SENSE=3 \
    PYTHONPATH=/ASR_FastAPI_WS_RU \
    DIAR_MODEL_OUTPUT_PATH=DIARISATION_model \
    DO_LOCAL_FILE_RECOGNITIONS=0 \
    DELETE_LOCAL_FILE_AFTR_ASR=0 \
    PYTHONPATH=/ASR_FastAPI_WS_RU \
    MAKE_MONO=0 \
    BETWEEN_WORDS_PERCENTILE=87
    # PROVIDER=CPU \
    # MODEL_NAME=gigaam-v3-ctc\

VOLUME ["/ASR_FastAPI_WS_RU/logs", \
       "/ASR_FastAPI_WS_RU/local_asr/to_asr", \
       "/ASR_FastAPI_WS_RU/models/hub", \
       "/ASR_FastAPI_WS_RU/local_asr/after_asr"]

WORKDIR /ASR_FastAPI_WS_RU

# Открываем порт и запускаем приложение
EXPOSE 49153
# На старте применяем миграции БД (идемпотентно). Если БД недоступна — не блокируем ASR.
CMD ["sh", "-c", "alembic upgrade head || echo 'alembic upgrade skipped'; exec python3 main.py"]