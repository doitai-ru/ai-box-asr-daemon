#!/usr/bin/env bash
# Локальный запуск ASR-сервера для разработки.
# Кодирует пути к CUDA-либам из venv (pip nvidia-* колёса) и нужные env-флаги.
#
# Использование:
#   ./run_local.sh                 # GPU (CUDA) + потоковый T-one на /ws
#   PROVIDER=CPU ./run_local.sh    # на CPU
#   USE_TONE_STREAMING=0 ./run_local.sh  # старый офлайн-обработчик на /ws
set -e
cd "$(dirname "$0")"

VENV="${VENV:-./venv}"
NV="$PWD/${VENV#./}/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$NV/cuda_nvrtc/lib:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/curand/lib:$NV/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

export HF_HOME="${HF_HOME:-./models}"
export USE_TONE_STREAMING="${USE_TONE_STREAMING:-1}"
export PROVIDER="${PROVIDER:-CUDA}"
export PORT="${PORT:-49155}"
export LOGGING_LEVEL="${LOGGING_LEVEL:-INFO}"
export DO_LOCAL_FILE_RECOGNITIONS="${DO_LOCAL_FILE_RECOGNITIONS:-0}"
export CAN_DIAR="${CAN_DIAR:-0}"

echo "PROVIDER=$PROVIDER USE_TONE_STREAMING=$USE_TONE_STREAMING PORT=$PORT"
exec "$VENV/bin/python" main.py
