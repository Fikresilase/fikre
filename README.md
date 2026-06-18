# fikre — Zindi Multilingual Health QA (Spark RAG Harness)

Portable **recipe** (not a prebuilt image) for running the RAG calibration harness on a
**DGX Spark (GB10, ARM64 + Blackwell, 128GB unified memory)**.

The image holds only the *environment*. Data, models, and outputs are **mounted volumes** — so the
image stays small, models download once, and results persist on the host.

```
fikre/
├── Dockerfile            # FROM NVIDIA NGC pytorch (arm64) + pip deps
├── docker-compose.yml    # GPU reservation + volume mounts + run command
├── requirements.txt
├── code/                 # the harness (Phases 0-6)
│   ├── config.py         # paths, model routing, retrieval/gen constants, seeds
│   ├── utils.py
│   ├── phase1_sample.py  # Val subset matched to Test per-subset counts
│   ├── phase2_index.py   # bge-m3 dense index over Train
│   ├── phase3_retrieve.py# dense retrieve top-20 per row
│   ├── phase4_rerank.py  # bge-reranker-v2-m3 -> top-8 exemplars
│   ├── phase5_generate.py# 3 model passes, full teardown between, checkpoints
│   ├── phase6_score.py   # ROUGE-1/L per row/subset/overall
│   └── run_all.py        # runs phases in order
├── data/                 # train.csv / val.csv / test.csv  (rsync here; NOT baked)
├── artifacts/            # outputs land here  (NOT baked)
└── hf-cache/             # models download here once (~126GB; NOT baked)
```

## Why "recipe, not image"

- **ARM64**: build on the Spark (`docker compose up --build`), not on an x86 laptop.
- **GPU/CUDA**: base image is the NGC PyTorch container already proven on the GB10.
- **126GB models / data / outputs**: mounted volumes, never inside the image.

## Run it (on the Spark, from `~/fikre`)

```bash
# 0. send data up from your laptop first:
#    rsync -av ./data/ milkesa@spark-9f1c:~/fikre/data/

# 1. build + run the whole harness:
docker compose up --build
# -> artifacts/scores_table.csv  (+ per-subset / overall printed)
```

First run downloads the 3 models into `hf-cache/` (slow, once). Later runs reuse them and resume
from cached artifacts after any crash.

## Serving backend

Generation defaults to **vLLM** (`GEN_BACKEND=vllm`). If vLLM does not build/run on the GB10, set
`GEN_BACKEND=hf` in `docker-compose.yml` to use the HuggingFace `transformers` fallback. Confirm with
the Stage C smoke test before the full run:

```bash
docker compose run --rm harness python /workspace/code/smoke_test.py
```

## IMPORTANT — verify before the full run

The three model repo IDs in `config.py` are **placeholders to confirm on HuggingFace**:
`Qwen3.5-35B-A3B`, `AfriqueQwen-14B`, `Sunflower-14B`. Fix the exact IDs there before downloading.
