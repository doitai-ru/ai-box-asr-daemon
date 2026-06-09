# Перевод прода ASR с Docker на нативную systemd-службу

Хост: `192.168.100.29` (ubuntu-nvidia, GTX 1080 8 ГБ). Папка проекта: `/opt/asr-amulex-it`.

## Что меняется

| | Было (Docker) | Стало (служба) |
|---|---|---|
| Запуск | контейнер `asr-amulex` (`asr-amulex:latest`, runtime nvidia) | `systemd` unit `asr-amulex.service` |
| Код | запечён в образ `/ASR_FastAPI_WS_RU` | `/opt/asr-amulex-it` (git, ветка `main`) |
| venv | в образе | `/opt/asr-amulex-it/venv` (перемещён из Vosk + доставлены T-one/DB) |
| Модели | bind-mount `/opt/ASR_docker/models` + `tone` в образе | перемещены в `/opt/asr-amulex-it/models` (+ `tone` из контейнера) |
| Логи | `/opt/asr-amulex/logs` | `/opt/asr-amulex-it/logs` |
| Профиль | — | `GPU_PROFILE=1` (`logs/gpu_profile.jsonl`) |

Прод-конфиг (env) перенесён в `deploy/asr-amulex.service` без изменений
(`PROVIDER=CUDA, MODEL_NAME=gigaam-v3-rnnt, STREAM_WITH_GPU=1, CAN_DIAR=1, DIAR_WITH_GPU=1,
PUNCTUATE_WITH_GPU=1, MAX_OVERLAP_DURATION=25, BETWEEN_WORDS_PERCENTILE=82, IS_PROD=1`),
плюс `HF_HOME=/opt/asr-amulex-it/models`, `TONE_DECODER=beam_search`, `GPU_PROFILE=1`.

## Порядок

### 1. Подготовка (один раз)
```bash
sudo bash /opt/asr-amulex-it/deploy/setup_native.sh
```
Гасит `vosk_gpu`, перемещает venv и модели, доставляет T-one/DB-зависимости, тянет `tone`
из контейнера, ставит и `enable`-ит службу (без старта).

### 2. Обновить код до main (после merge ветки профиля в main)
```bash
git -C /opt/asr-amulex-it pull
```
Нужно, чтобы на проде появился код профилировщика (иначе `GPU_PROFILE=1` просто игнорится).

### 3. Cutover
```bash
docker stop asr-amulex            # освободить GPU и порт 49153
sudo systemctl start asr-amulex
journalctl -u asr-amulex -f       # дождаться прогрева моделей + "GPU-профилировщик включён"
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:49153/is_alive   # 200
tail -2 /opt/asr-amulex-it/logs/gpu_profile.jsonl                          # профиль пишется
```

### 4. Удалить Docker-образ (после успешной проверки)
```bash
docker rm asr-amulex
docker rmi asr-amulex:latest
```

## Откат
```bash
sudo systemctl stop asr-amulex
docker start asr-amulex
```
(Пока образ не удалён — откат мгновенный. Модели/venv перемещены, но контейнеру они не нужны:
он несёт свои в образе/`tone` запечён, модели бил bind-mount'ил из `/opt/ASR_docker/models` —
после `mv` этот путь пуст, поэтому для отката контейнера сначала верните модели обратно
или не удаляйте `/opt/ASR_docker/models` до подтверждения работы службы.)
