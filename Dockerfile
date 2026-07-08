FROM python:3.11-slim

WORKDIR /app

# System deps for llama-cpp-python compilation + curl for model download
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download local model (Q4 quantisation — ~350MB, quality is solid for sentiment/factual/summarisation)
RUN mkdir -p /app/models && \
    curl -L -o /app/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    "https://huggingface.co/TheBloke/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2_5-0.5b-instruct-q4_k_m.gguf"

COPY main.py /app/main.py
RUN mkdir -p /input /output

# Harness injects at runtime: FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS
ENTRYPOINT ["python", "/app/main.py"]
