# Base: NVIDIA NGC PyTorch — arch-correct for ARM64 + Blackwell (GB10).
# Pin to the tag you proved in Stage C (`torch.cuda.get_device_name(0)` -> NVIDIA GB10).
ARG NGC_TAG=25.05-py3
FROM nvcr.io/nvidia/pytorch:${NGC_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /workspace

# Lightweight, arch-independent deps first (cached layer).
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt hf_transfer

# vLLM installed separately so a torch resolution conflict can't poison the rest.
# If vLLM has no GB10 wheel and tries to build from source and fails, set GEN_BACKEND=hf
# in docker-compose.yml and comment this line out — the harness falls back to transformers.
RUN pip install --no-cache-dir vllm || \
    echo "WARNING: vllm install failed — set GEN_BACKEND=hf to use the transformers fallback."

# Code is also mounted as a volume at runtime; copying it in keeps the image self-contained too.
COPY code/ /workspace/code/

CMD ["python", "/workspace/code/run_all.py"]
