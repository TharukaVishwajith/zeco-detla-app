FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_ENV=production \
    LOG_LEVEL=INFO \
    OPENAI_CHAT_MODEL=gpt-5.2 \
    OPENAI_EMBEDDING_MODEL=text-embedding-3-small \
    OPENAI_EMBEDDING_DIMENSIONS=1536 \
    OPENSEARCH_INDEX=knowledge_base \
    OPENSEARCH_REGION=us-east-1 \
    OPENSEARCH_PORT=443 \
    OPENSEARCH_VERIFY_CERTS=true \
    OPENSEARCH_VECTOR_FIELD=embeddings \
    TICKET_API_TIMEOUT_SECONDS=10 \
    RETRIEVAL_TOP_K=10 \
    DYNAMODB_REGION=ap-southeast-2 \
    CONVERSATION_HISTORY_MAX_MESSAGES=12 \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000 \
    UVICORN_RELOAD=false

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN groupadd --system appuser \
    && useradd --system --gid appuser --create-home appuser \
    && chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"UVICORN_PORT\", \"8000\")}/health', timeout=10)"

EXPOSE 8000

CMD ["python", "main.py"]
