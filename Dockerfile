# Base: NVIDIA NGC PyTorch — arch-correct for ARM64 + Blackwell (GB10).
# Pin to the tag you proved in Stage C (`torch.cuda.get_device_name(0)` -> NVIDIA GB10).
ARG NGC_TAG=25.05-py3
FROM nvcr.io/nvidia/pytorch:${NGC_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /workspace

# CRITICAL: the NGC image ships a custom CUDA-enabled torch. A plain `pip install` of
# anything depending on torch will REPLACE it with a CPU-only PyPI wheel and kill CUDA.
# So we pin torch to the already-installed version via a constraints file — pip then sees
# the requirement satisfied and never touches torch.
COPY requirements.txt /tmp/requirements.txt
RUN PINNED_TORCH="$(python -c 'import torch; print(torch.__version__)')" && \
    echo "torch==${PINNED_TORCH}" > /tmp/constraints.txt && \
    echo "[build] protecting torch==${PINNED_TORCH}" && \
    PIP_CONSTRAINT=/tmp/constraints.txt \
        pip install --no-cache-dir -r /tmp/requirements.txt hf_transfer && \
    python -c "import torch, sys; v=torch.__version__; \
               sys.exit('torch was swapped to a CPU wheel: '+v) if '+cpu' in v else \
               print('[build] torch preserved:', v)"

# vLLM is intentionally NOT installed: its PyPI wheel pulled torch 2.9.0+cpu and broke CUDA
# on this ARM/Blackwell image. The harness uses the HF transformers backend (GEN_BACKEND=hf),
# which runs on the NGC torch. (To experiment with vLLM later, do it in a separate image.)

# Code is also mounted as a volume at runtime; copying it in keeps the image self-contained too.
COPY code/ /workspace/code/

CMD ["python", "/workspace/code/run_all.py"]
