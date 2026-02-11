# Настройка сервера для перевода книги

## Выбор модели

На 8xH100 80GB (640GB VRAM суммарно) можно запустить:

### Вариант 1: Llama 3.1 405B FP8 (РЕКОМЕНДУЕТСЯ)
- **Размер**: ~405GB в FP8 — идеально помещается в 640GB
- **Качество**: лучшая dense-модель для литературного перевода EN→RU
- **Скорость**: ~30-50 tok/s при batch=1
- **Модель**: `neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8`

### Вариант 2: DeepSeek-V3 671B MoE (FP8)
- **Размер**: ~671B параметров, но MoE (37B активных) — тесно на 640GB, но возможно
- **Качество**: потенциально выше 405B за счёт масштаба
- **Скорость**: быстрее 405B при инференсе (MoE)
- **Модель**: `deepseek-ai/DeepSeek-V3`
- **Нюанс**: vLLM рекомендует 8xH200, на H100 будет мало KV-кэша

### Вариант 3: Qwen2.5-72B-Instruct (если хотите быстрее)
- **Размер**: ~72GB в FP8 — огромный запас памяти
- **Качество**: 95% качества 405B, отличный русский
- **Скорость**: ~150-200 tok/s — в 4x быстрее 405B
- **Модель**: `Qwen/Qwen2.5-72B-Instruct`

---

## Полная инструкция установки

### Шаг 1: Подготовка окружения

```bash
# Создать conda/venv окружение
conda create -n translate python=3.11 -y
conda activate translate

# Установить vLLM (включает PyTorch с CUDA)
pip install vllm

# Установить зависимости скрипта перевода
pip install PyMuPDF fpdf2 openai tqdm
```

### Шаг 2: Скачать модель (делается один раз)

```bash
# Установить huggingface-cli если ещё нет
pip install huggingface_hub[cli]

# --- Для Llama 405B FP8 (рекомендуется) ---
# Нужно принять лицензию на https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct
# Затем:
huggingface-cli login  # ввести токен с huggingface.co/settings/tokens
huggingface-cli download neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8 \
    --local-dir /models/llama-405b-fp8

# --- ИЛИ для DeepSeek-V3 ---
huggingface-cli download deepseek-ai/DeepSeek-V3 \
    --local-dir /models/deepseek-v3

# --- ИЛИ для Qwen 72B ---
huggingface-cli download Qwen/Qwen2.5-72B-Instruct \
    --local-dir /models/qwen-72b
```

### Шаг 3: Запуск vLLM сервера

#### Llama 3.1 405B FP8 (рекомендуется)

```bash
vllm serve neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8 \
    --tensor-parallel-size 8 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.92 \
    --port 8000 \
    --host 0.0.0.0
```

Или если скачали локально:

```bash
vllm serve /models/llama-405b-fp8 \
    --tensor-parallel-size 8 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.92 \
    --port 8000 \
    --host 0.0.0.0
```

#### DeepSeek-V3 671B MoE

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --trust-remote-code \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.95 \
    --port 8000 \
    --host 0.0.0.0
```

#### Qwen2.5-72B-Instruct (быстрый вариант)

```bash
vllm serve Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 8 \
    --quantization fp8 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --port 8000 \
    --host 0.0.0.0
```

### Шаг 4: Проверить что сервер работает

```bash
# В другом терминале:
curl http://localhost:8000/v1/models

# Тестовый запрос:
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8",
        "messages": [
            {"role": "system", "content": "Translate to Russian."},
            {"role": "user", "content": "The sun was setting over Dubai."}
        ],
        "temperature": 0.3,
        "max_tokens": 256
    }'
```

### Шаг 5: Запуск перевода книги

```bash
# Скопировать файлы на сервер (scp, rsync и т.д.)
scp translate_pdf.py requirements.txt book.pdf user@server:/path/to/work/

# На сервере:
cd /path/to/work

# Для Llama 405B:
python translate_pdf.py \
    --input "_OceanofPDF.com_My_Story_-_Mohammed_bin_rashid.pdf" \
    --output "My_Story_translated.pdf" \
    --api-url "http://localhost:8000/v1" \
    --model "neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8" \
    --batch-size 3 \
    --temperature 0.3

# Для DeepSeek-V3:
python translate_pdf.py \
    --input "_OceanofPDF.com_My_Story_-_Mohammed_bin_rashid.pdf" \
    --output "My_Story_translated.pdf" \
    --api-url "http://localhost:8000/v1" \
    --model "deepseek-ai/DeepSeek-V3" \
    --batch-size 5 \
    --temperature 0.3

# Для Qwen 72B:
python translate_pdf.py \
    --input "_OceanofPDF.com_My_Story_-_Mohammed_bin_rashid.pdf" \
    --output "My_Story_translated.pdf" \
    --api-url "http://localhost:8000/v1" \
    --model "Qwen/Qwen2.5-72B-Instruct" \
    --batch-size 5 \
    --temperature 0.3
```

---

## Оценка времени перевода

Книга: ~1000 абзацев, ~150K слов

| Модель | tok/s (batch=1) | Примерное время |
|--------|-----------------|-----------------|
| Llama 405B FP8 | 30-50 | 40-60 мин |
| DeepSeek-V3 MoE | 50-80 | 25-40 мин |
| Qwen 72B FP8 | 150-200 | 10-15 мин |

При батчинге (3-5 абзацев за запрос) время сокращается дополнительно.

---

## Советы

1. **batch-size**: для 405B лучше 3 (длинные ответы), для 72B можно 5
2. **temperature**: 0.3 для стабильного перевода, 0.5 для более "живого"
3. **Чекпоинты**: скрипт сохраняет прогресс после каждого батча — если
   процесс прервётся, просто запустите ту же команду снова
4. **max-model-len**: 4096 достаточно для перевода абзацев; увеличивать
   нет смысла (займёт больше VRAM под KV-кэш)
5. **Мониторинг GPU**: `watch -n1 nvidia-smi` в отдельном терминале

## Запуск в Docker (альтернатива)

```bash
docker run --runtime nvidia --gpus all \
    -v /models:/models \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model /models/llama-405b-fp8 \
    --tensor-parallel-size 8 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.92
```
