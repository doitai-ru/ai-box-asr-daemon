#!/usr/bin/env bash
# Поднять ПАРАЛЛЕЛЬНЫЙ T-one-контейнер рядом с продом, НЕ трогая текущий (asr на 49153).
#
# Запускать на хосте 192.168.100.29 из корня репозитория (где лежит Docker/Dockerfile).
# ВАЖНО: перед сборкой запушить T-one правки в gl.amulex.ru/amulex/ml-ai/asr (ветка $GIT_REF),
#        иначе образ соберётся без потокового пути /ws-stream.
#
# Использование:
#   GIT_TOKEN=<deploy/CI токен с read_repository> ./Docker/run_asr_tone.sh
set -e

GIT_TOKEN="${GIT_TOKEN:?Задай GIT_TOKEN (deploy/CI токен gl.amulex.ru с правом read_repository)}"
GIT_REF="${GIT_REF:-main}"
IMAGE="${IMAGE:-asr-tone}"
NAME="${NAME:-asr-tone}"
HOST_PORT="${HOST_PORT:-49154}"        # хост-порт; внутри контейнера приложение слушает 49153
DEPLOY_DIR="${DEPLOY_DIR:-/opt/asr_tone}"
PROD_HUB="${PROD_HUB:-/opt/ASR_docker/models/hub}"  # переиспользуем кэш моделей прода (GigaAM уже там; T-one дольёт свои)

# каталог logs обязателен (иначе TimedRotatingFileHandler падает на старте)
sudo mkdir -p "$DEPLOY_DIR/logs"

# 1) Сборка из НАШЕГО форка. PROVIDER=CUDA -> onnxruntime-gpu в образе.
#    Если gl.amulex.ru доступен только по SSH, замени HTTPS+token на deploy-key в Dockerfile.
docker build -f Docker/Dockerfile \
  --build-arg GIT_TOKEN="$GIT_TOKEN" \
  --build-arg GIT_REF="$GIT_REF" \
  --build-arg PROVIDER=CUDA \
  -t "$IMAGE" .

# 2) (Пере)запуск второго контейнера. USE_TONE_STREAMING НЕ задаём ->
#    обе ручки параллельно: /ws (офлайн GigaAM) и /ws-stream (T-one).
docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" --restart unless-stopped \
  --runtime nvidia --gpus all \
  -p "${HOST_PORT}:49153" \
  -e PROVIDER=CUDA \
  -e MODEL_NAME=gigaam-v3-rnnt \
  -e BASE_SAMPLE_RATE=16000 \
  -e MAX_OVERLAP_DURATION=25 \
  -e CAN_DIAR=1 \
  -e DIAR_WITH_GPU=1 \
  -e TONE_DECODER=beam_search \
  -v "$PROD_HUB":/ASR_FastAPI_WS_RU/models/hub \
  -v "$DEPLOY_DIR/logs":/ASR_FastAPI_WS_RU/logs \
  "$IMAGE"

echo "OK: $NAME поднят."
echo "  офлайн GigaAM : ws://<host>:${HOST_PORT}/ws"
echo "  T-one стрим   : ws://<host>:${HOST_PORT}/ws-stream"
echo "Внимание: 2x GigaAM-v3-rnnt на GPU 8 ГБ может не влезть. Если OOM -"
echo "  для tone-контейнера поставь CAN_DIAR=0 или PROVIDER=CPU (T-one всё равно на CPU)."
