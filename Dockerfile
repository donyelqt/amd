FROM python:3.11-slim

WORKDIR /app

COPY main.py /app/main.py
COPY .env.example /app/.env.example

RUN pip install --no-cache-dir python-dotenv && \
    mkdir -p /input /output

# Harness injects at runtime: FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS
ENTRYPOINT ["python", "/app/main.py"]
