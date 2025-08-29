# База с PyTorch 2.2.2 + CUDA 11.8 (совместима с T4/A10/A100)
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    HF_HOME=/models \
    TRANSFORMERS_CACHE=/models \
    HUGGINGFACE_HUB_CACHE=/models \
    MODEL_ID=LanguageBind/MoE-LLaVA-Phi2-2.7B-4e-384 \
    LOAD_4BIT=false \
    LOAD_8BIT=false \
    PRECISION=fp16 \
    MAX_NEW_TOKENS=256 \
    TOKENIZERS_PARALLELISM=false

# ---------- Системные зависимости ДО pip install ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs libgl1 ca-certificates curl && \
    rm -rf /var/lib/apt/lists/* && git lfs install

WORKDIR /app

# ---------- Python зависимости (пинним стабильный стек) ----------
COPY src/requirements.txt /app/src/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel packaging && \
    pip install --no-cache-dir -r /app/src/requirements.txt

# Ставим MoE-LLaVA из Git отдельно, БЕЗ зависимостей — чтобы не ломать пины
# Можно задать конкретный тег/коммит через ARG
ARG MOELLAVA_REF=v1.0.0
RUN pip install --no-cache-dir --no-deps \
    "git+https://github.com/PKU-YuanGroup/MoE-LLaVA.git"

# Код
COPY src /app/src

# (опционально) прогрев весов в volume, чтобы сократить холодный старт
# RUN python /app/src/download_models.py || true

CMD ["python", "-u", "/app/src/rp_handler.py"]
