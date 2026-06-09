#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# Установка нативной systemd-службы ASR на хосте С НУЛЯ (нативный порт Dockerfile).
# Запускать на хосте от root (sudo). Идемпотентно: уже сделанное пропускается.
#
#   sudo bash /opt/asr-amulex-it/deploy/setup_native.sh
#
# Требования на хосте: nvidia-драйвер (CUDA), git, доступ в сеть (pip/HF/модели),
# системный python3.12 СО встроенным sqlite (обычно /usr/bin/python3.12).
#
# Делает: системные зависимости -> свежий venv (onnxruntime-gpu + T-one + cupy) ->
# cuDNN-пин под Pascal -> модели (если нет — докачивает как Dockerfile) -> служба.
# Cutover/удаление Docker — отдельно, см. deploy/README.md.

set -euo pipefail

REPO=/opt/asr-amulex-it
VENV=$REPO/venv
PYBASE=${PYBASE:-/usr/bin/python3.12}          # питон с sqlite (НЕ /usr/local, он без _sqlite3)
ORT_GPU_VERSION=${ORT_GPU_VERSION:-1.23.2}      # как в прод-образе
TONE_REF=3c5b6c015038173840e62cea99e10cdb1c759116
DIAR_MODEL_NAME=${DIAR_MODEL_NAME:-voxblink2_samresnet100_ft}
TONE_DECODER=${TONE_DECODER:-beam_search}
DIAR_API_URL=https://modelscope.cn/api/v1/datasets/wenet/wespeaker_pretrained_models/oss/tree

echo "== 1) Системные зависимости (терпим битые сторонние репозитории) =="
apt-get update -y || echo "apt-get update: часть репозиториев недоступна — продолжаем"
apt-get install -y git git-lfs ffmpeg curl ca-certificates jq \
    build-essential cmake libboost-all-dev libeigen3-dev \
    python3.12 python3.12-venv

echo "== 2) Свежий venv на питоне с sqlite =="
if ! "$PYBASE" -c "import sqlite3" 2>/dev/null; then
    echo "ОШИБКА: $PYBASE собран без sqlite (_sqlite3). Укажите PYBASE=<питон со sqlite> или поставьте python3.12 из дистрибутива."
    exit 1
fi
rm -rf "$VENV"
"$PYBASE" -m venv "$VENV"
"$VENV/bin/python" -m pip install --no-cache-dir -U pip

echo "== 3) Зависимости (onnxruntime-gpu + cupy, как в Dockerfile для CUDA) =="
REQ=$(mktemp)
cp "$REPO/requirements.txt" "$REQ"
sed -i "s#^onnxruntime\\s*\$#onnxruntime-gpu[cuda,cudnn]==${ORT_GPU_VERSION}#" "$REQ"
echo "cupy-cuda12x==13.5.1" >> "$REQ"
"$VENV/bin/python" -m pip install --no-cache-dir -r "$REQ"
rm -f "$REQ"

echo "== 4) T-one стек (запинено под прод-образ; --no-deps, чтобы не тянуть CPU-onnxruntime) =="
"$VENV/bin/python" -m pip install --no-cache-dir --no-deps \
    "git+https://github.com/voicekit-team/T-one.git@${TONE_REF}"
"$VENV/bin/python" -m pip install --no-cache-dir "pyctcdecode==0.5.0" "kenlm==0.3.0" "miniaudio==1.71"

echo "== 5) cuDNN под Pascal (GTX 10xx): RNN на cudnn 9.20 падает, нужен 9.6.0.74 =="
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
echo "  GPU: ${GPU_NAME:-неизвестно}"
case "$GPU_NAME" in
    *GTX\ 10*|*GTX\ 9*|*Pascal*|*P100*|*P40*)
        echo "  Pascal -> пиним nvidia-cudnn-cu12==9.6.0.74"
        "$VENV/bin/python" -m pip install --no-cache-dir --force-reinstall --no-deps "nvidia-cudnn-cu12==9.6.0.74"
        ;;
    *)
        echo "  не Pascal — оставляем текущий cuDNN"
        ;;
esac

echo "== 6) Проверка стека =="
"$VENV/bin/python" - <<'PY'
import sqlite3, soxr, tone, kenlm, pydantic_settings  # noqa
import onnxruntime as o
print("onnxruntime", o.__version__, o.get_available_providers())
print("sqlite", sqlite3.sqlite_version, "+ tone/kenlm/soxr/pydantic_settings OK")
PY

echo "== 7) Модели (докачиваем недостающее в {repo}/models, как Dockerfile) =="
M="$REPO/models"
mkdir -p "$M"
git lfs install 2>/dev/null || true
# пунктуация
if [ ! -e "$M/sbert_punc_case_ru_onnx/config.json" ] && [ ! -e "$M/sbert_punc_case_ru_onnx" ]; then
    git clone https://huggingface.co/Alexanrd/sbert_punc_case_ru_onnx "$M/sbert_punc_case_ru_onnx"
else echo "  sbert_punc_case_ru_onnx — на месте"; fi
# VAD
if [ ! -e "$M/VAD_silero_v5/silero_vad.onnx" ]; then
    mkdir -p "$M/VAD_silero_v5"
    curl -fL https://github.com/snakers4/silero-vad/raw/v5.0/files/silero_vad.onnx -o "$M/VAD_silero_v5/silero_vad.onnx"
else echo "  VAD — на месте"; fi
# Диаризация
if [ ! -e "$M/DIARISATION_model/${DIAR_MODEL_NAME}.onnx" ]; then
    mkdir -p "$M/DIARISATION_model"
    curl -s -L -H "User-Agent: Mozilla/5.0" "$DIAR_API_URL" > /tmp/diar_api.json
    url=$(jq -r --arg m "${DIAR_MODEL_NAME}.onnx" '.Data[] | select(.Key==$m) | .Url' /tmp/diar_api.json)
    [ -n "$url" ] && curl -L "$url" -o "$M/DIARISATION_model/${DIAR_MODEL_NAME}.onnx" || echo "  ВНИМАНИЕ: модель диаризации $DIAR_MODEL_NAME не найдена в API"
    rm -f /tmp/diar_api.json
else echo "  DIARISATION_model — на месте"; fi
# T-one (в models/tone; иначе подтянется из HF-кэша hub при старте)
if [ ! -e "$M/tone/model.onnx" ] && ! ls -d "$M"/hub/*T-one* >/dev/null 2>&1; then
    HF_HOME=/tmp/hfcache "$VENV/bin/python" -c "import os; from tone import StreamingCTCPipeline; d='$M/tone'; os.makedirs(d, exist_ok=True); StreamingCTCPipeline.download_from_hugging_face(d, only_acoustic=('$TONE_DECODER'=='greedy'))"
    rm -rf /tmp/hfcache
else echo "  T-one — на месте (models/tone или hub)"; fi
# GigaAM (gigaam-v3-*) подтянется onnx_asr в HF-кэш hub при первом старте.

echo "== 8) Лог-каталог + служба =="
mkdir -p "$REPO/logs"
install -m 0644 "$REPO/deploy/asr-amulex.service" /etc/systemd/system/asr-amulex.service
systemctl daemon-reload
systemctl enable asr-amulex

echo
echo "ГОТОВО. Старт:"
echo "  systemctl start asr-amulex && journalctl -u asr-amulex -f"
echo "  curl -s -o /dev/null -w '%{http_code}\\n' http://127.0.0.1:49153/is_alive   # 200"
