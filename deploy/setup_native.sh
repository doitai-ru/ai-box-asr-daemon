#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# Подготовка нативной systemd-службы ASR на проде (перевод с Docker).
# Запускать на хосте 192.168.100.29 от root (sudo). Идемпотентно по возможности.
#
#   sudo bash /opt/asr-amulex-it/deploy/setup_native.sh
#
# Что делает (cutover — отдельно, см. deploy/README.md):
#   1) гасит старую службу vosk_gpu;
#   2) МУВИТ venv из /opt/Vosk5_FastAPI_streaming -> /opt/asr-amulex-it (мало места — не копируем);
#   3) доставляет недостающие зависимости (DB-слой, pydantic-settings) + T-one стек;
#   4) МУВИТ модели из /opt/ASR_docker/models -> /opt/asr-amulex-it/models;
#   5) тянет tone из работающего Docker-контейнера (он запечён в образ, не в bind-mount);
#   6) ставит и включает (без старта) службу asr-amulex.

set -euo pipefail

REPO=/opt/asr-amulex-it
VENV=$REPO/venv
VOSK_VENV=/opt/Vosk5_FastAPI_streaming/venv
DOCKER_MODELS=/opt/ASR_docker/models
CONTAINER=asr-amulex

echo "== 1) Гасим старую службу vosk_gpu =="
systemctl stop vosk_gpu 2>/dev/null || true
systemctl disable vosk_gpu 2>/dev/null || true

echo "== 2) Перемещаем venv (мало места — mv, не cp) =="
if [ -x "$VENV/bin/python" ]; then
    echo "venv уже на месте: $VENV"
elif [ -x "$VOSK_VENV/bin/python" ]; then
    mv "$VOSK_VENV" "$VENV"
    echo "venv перемещён: $VOSK_VENV -> $VENV"
else
    echo "ОШИБКА: не найден ни $VENV, ни $VOSK_VENV"; exit 1
fi

echo "== 3) Сборочные зависимости для kenlm =="
# Терпим битые/недоступные сторонние репозитории (напр. устаревший локальный CUDA-репо):
# нужные пакеты идут из штатных ubuntu-репозиториев, которые обновляются нормально.
apt-get update -y || echo "apt-get update: часть репозиториев недоступна — продолжаем"
apt-get install -y build-essential cmake libboost-all-dev libeigen3-dev ffmpeg

echo "== 3b) Доставляем наши зависимости поверх (onnxruntime-gpu не трогаем) =="
"$VENV/bin/python" -m pip install --no-cache-dir -r "$REPO/requirements.txt"
echo "== 3c) T-one стек (--no-deps, чтобы не затянуть CPU-onnxruntime поверх GPU) =="
# Версии запинены под те, что стоят в прод-образе asr-amulex (supply chain: фикс на
# конкретный коммит T-one + точные версии пакетов; заодно точное воспроизведение прода).
TONE_REF=3c5b6c015038173840e62cea99e10cdb1c759116
"$VENV/bin/python" -m pip install --no-cache-dir --no-deps \
    "git+https://github.com/voicekit-team/T-one.git@${TONE_REF}"
"$VENV/bin/python" -m pip install --no-cache-dir "pyctcdecode==0.5.0" "kenlm==0.3.0" "miniaudio==1.71"

echo "== 3d) Проверка стека =="
"$VENV/bin/python" - <<'PY'
import onnxruntime as o
import tone, kenlm, sqlalchemy, pydantic_settings, soxr  # noqa
print("onnxruntime", o.__version__, o.get_available_providers())
print("tone/kenlm/sqlalchemy/pydantic_settings/soxr OK")
PY

echo "== 4) Перемещаем модели -> {repo}/models (mv) =="
mkdir -p "$REPO/models"
if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; then
    echo "  контейнер $CONTAINER ещё РАБОТАЕТ (использует bind-mount моделей)."
    echo "  Модели НЕ трогаю — переместить на cutover ПОСЛЕ 'docker stop $CONTAINER':"
    echo "    for d in hub VAD_silero_v5 sbert_punc_case_ru_onnx DIARISATION_model; do mv $DOCKER_MODELS/\$d $REPO/models/\$d; done"
else
    for d in hub VAD_silero_v5 sbert_punc_case_ru_onnx DIARISATION_model; do
        if [ -e "$REPO/models/$d" ]; then
            echo "  $d уже в repo/models — пропускаю"
        elif [ -e "$DOCKER_MODELS/$d" ]; then
            mv "$DOCKER_MODELS/$d" "$REPO/models/$d"
            echo "  перемещён: $d"
        else
            echo "  ВНИМАНИЕ: $d не найден в $DOCKER_MODELS"
        fi
    done
fi

echo "== 5) tone из контейнера (запечён в образ) =="
if [ -e "$REPO/models/tone/model.onnx" ]; then
    echo "  tone уже на месте"
elif docker inspect "$CONTAINER" >/dev/null 2>&1; then
    docker cp "$CONTAINER:/ASR_FastAPI_WS_RU/models/tone" "$REPO/models/tone"
    echo "  tone скопирован из контейнера"
else
    echo "  ВНИМАНИЕ: контейнер $CONTAINER недоступен — tone не скопирован (скачается из HF при старте)"
fi

echo "== 5b) Лог-каталог =="
mkdir -p "$REPO/logs"

echo "== 6) Установка службы (enable, без start) =="
install -m 0644 "$REPO/deploy/asr-amulex.service" /etc/systemd/system/asr-amulex.service
systemctl daemon-reload
systemctl enable asr-amulex

echo
echo "ГОТОВО (подготовка). Cutover — вручную (см. deploy/README.md):"
echo "  git -C $REPO pull            # после merge ветки в main (профиль + service)"
echo "  docker stop $CONTAINER"
echo "  systemctl start asr-amulex"
echo "  curl -s -o /dev/null -w '%{http_code}\\n' http://127.0.0.1:49153/is_alive"
echo "  # после проверки: docker rm $CONTAINER && docker rmi asr-amulex:latest"
