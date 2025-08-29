FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    TRANSFORMERS_CACHE=/models \
    HUGGINGFACE_HUB_CACHE=/models \
    MODEL_ID=LanguageBind/MoE-LLaVA-Phi2-2.7B-4e-384 \
    LOAD_4BIT=false \
    LOAD_8BIT=false \
    PRECISION=fp16 \
    MAX_NEW_TOKENS=256 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs libgl1 ca-certificates curl && \
    rm -rf /var/lib/apt/lists/* && git lfs install

COPY src/requirements.txt /app/src/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/src/requirements.txt

# Код
COPY src /app/src


CMD ["python", "-u", "/app/src/rp_handler.py"]
