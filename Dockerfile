FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    PORT=8080 \
    HF_HOME=/tmp/huggingface \
    HF_HUB_CACHE=/tmp/huggingface/hub \
    HUGGINGFACE_HUB_CACHE=/tmp/huggingface/hub \
    SENTENCE_TRANSFORMERS_HOME=/tmp/huggingface/sentence-transformers \
    TORCH_HOME=/tmp/torch \
    XDG_CACHE_HOME=/tmp/.cache \
    PUPPETEER_CACHE_DIR=/tmp/puppeteer \
    NODE_ENV=production

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    wget \
    dumb-init \
    software-properties-common \
    build-essential \
    git \
    xdg-utils \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcairo2 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libexpat1 \
    libgbm1 \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxkbcommon0 \
    libxrandr2 \
    libxrender1 \
    libxshmfence1 \
    libxss1 \
    libxtst6 \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get update && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && python -m ensurepip --upgrade \
    && python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt /app/requirements.txt

RUN python -m pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 \
    torchvision==0.23.0

RUN python -m pip install -r /app/requirements.txt

COPY multiview_worker/package*.json /app/multiview_worker/

RUN if [ -f /app/multiview_worker/package.json ]; then \
      npm --prefix /app/multiview_worker ci --omit=dev || npm --prefix /app/multiview_worker install --omit=dev; \
    fi

COPY . /app

RUN mkdir -p \
    /tmp/huggingface \
    /tmp/huggingface/hub \
    /tmp/huggingface/sentence-transformers \
    /tmp/torch \
    /tmp/.cache \
    /tmp/puppeteer \
    /app/tmp \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app /tmp/huggingface /tmp/torch /tmp/.cache /tmp/puppeteer

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8080\")}/health', timeout=3).read()" || exit 1

ENTRYPOINT ["dumb-init", "--"]

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers"]