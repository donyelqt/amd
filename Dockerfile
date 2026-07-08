FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/models && \
    curl -L -o /app/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2_5-0.5b-instruct-q4_k_m.gguf"

COPY main.py /app/main.py
RUN mkdir -p /input /output

ENTRYPOINT ["python", "/app/main.py"]
