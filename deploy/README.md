# Нативная systemd-служба ASR (вместо Docker)

Прод-хост: `192.168.100.29` (ubuntu-nvidia, GTX 1080 8 ГБ, Pascal). Папка: `/opt/asr-amulex-it`.
Служба `asr-amulex.service` запускает приложение нативно (без Docker): gigaam-v3-rnnt +
T-one (`/api/v1/asr/ws-stream`) + диаризация + пунктуация, GPU, профилировщик (`GPU_PROFILE=1`).

## Состав

| | Значение |
|---|---|
| Код | `/opt/asr-amulex-it` (git, ветка `main`) |
| venv | `/opt/asr-amulex-it/venv` (свежий, `onnxruntime-gpu==1.23.2` + T-one + cupy) |
| Модели | `/opt/asr-amulex-it/models` (hub: gigaam-v3 + T-one; VAD/sbert/DIARISATION) |
| Логи | `/opt/asr-amulex-it/logs` (+ профиль `logs/gpu_profile.jsonl`) |
| Конфиг | в `deploy/asr-amulex.service` (env: PROVIDER=CUDA, MODEL_NAME=gigaam-v3-rnnt, STREAM_WITH_GPU=1, CAN_DIAR=1, DIAR_WITH_GPU=1, PUNCTUATE_WITH_GPU=1, MAX_OVERLAP_DURATION=25, IS_PROD=1, HF_HOME=…/models, TONE_DECODER=beam_search, GPU_PROFILE=1) |

## Установка с нуля

Требуется: nvidia-драйвер (CUDA), сеть (pip/HF), системный `python3.12` со встроенным
sqlite (обычно `/usr/bin/python3.12`; `/usr/local`-питон, собранный из исходников, часто
БЕЗ `_sqlite3` — тогда укажите `PYBASE=/usr/bin/python3.12`).

```bash
sudo bash /opt/asr-amulex-it/deploy/setup_native.sh
```

Скрипт (нативный порт `Docker/Dockerfile`): системные зависимости (+build-tools для kenlm),
**свежий** venv на питоне с sqlite, `onnxruntime-gpu` + T-one (запинено), **пин cuDNN
`9.6.0.74` под Pascal**, докачка недостающих моделей (sbert/VAD/DIARISATION; gigaam и T-one —
из HF при первом старте/в hub), установка и `enable` службы.

## Старт и проверка

```bash
sudo systemctl start asr-amulex
journalctl -u asr-amulex -f      # прогрев моделей, "GPU-профилировщик включён", Application startup complete
curl -s -o /dev/null -w 'is_alive: %{http_code}\n' http://127.0.0.1:49153/is_alive   # 200
tail -2 /opt/asr-amulex-it/logs/gpu_profile.jsonl                                     # профиль пишется
```

## Грабли (из реальной миграции на этом хосте)

- **Pascal (GTX 1080) + cuDNN.** RNN-операции gigaam-rnnt на `nvidia-cudnn-cu12` 9.20 падают
  (`CUDNN_STATUS_EXECUTION_FAILED_CUDART` на `cudnnSetDropoutDescriptor`). Нужен **9.6.0.74**
  (скрипт пинит автоматически при Pascal-GPU; форс: `pip install --force-reinstall --no-deps nvidia-cudnn-cu12==9.6.0.74`).
- **Питон без sqlite.** Если venv собрать на питоне без `_sqlite3` (часто `/usr/local`-сборка),
  api/v1 падает на `aiosqlite`→`sqlite3`. Скрипт создаёт venv на `python3.12` с sqlite.
- **T-one** грузится из HF-кэша `hub` (`models--t-tech--T-one`) через `from_hugging_face`, если
  нет `models/tone` — отдельная папка `tone` не обязательна.
- **CUDA-окружение.** onnxruntime-gpu несёт CUDA/cuDNN pip-колёсами и грузит их сам
  (`ort.preload_dlls`), отдельный `LD_LIBRARY_PATH` на системную CUDA не нужен.

## Управление

```bash
systemctl status asr-amulex
systemctl restart asr-amulex
journalctl -u asr-amulex -f
```
