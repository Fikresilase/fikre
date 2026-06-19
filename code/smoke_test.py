"""Stage C smoke test — run BEFORE the full pipeline.

Checks, in order:
  1. torch sees the GB10
  2. data files load with expected columns
  3. embedder + reranker load and run on a tiny input
  4. generation backend loads ONE model and produces one line

Usage (inside the container):
  python /workspace/code/smoke_test.py
  python /workspace/code/smoke_test.py --model qwen|afriqwen|sunflower
"""
import argparse

import pandas as pd

import config as C
from prompts import build_messages
from utils import free_vram


def check_torch():
    import torch
    print(f"[1] torch {torch.__version__} cuda={torch.cuda.is_available()} "
          f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
    assert torch.cuda.is_available(), "CUDA not available in container!"


def check_data():
    for name, path in [("train", C.TRAIN_CSV), ("val", C.VAL_CSV), ("test", C.TEST_CSV)]:
        df = pd.read_csv(path, nrows=5)
        print(f"[2] {name}: cols={list(df.columns)}")
        assert C.COL_ID in df.columns and C.COL_IN in df.columns, f"{name} missing ID/input"


def check_retrieval():
    from FlagEmbedding import BGEM3FlagModel
    from rerank_model import Reranker
    emb = BGEM3FlagModel(C.EMBEDDER, use_fp16=True)
    v = emb.encode(["What is malaria?", "Olumbe ki?"])["dense_vecs"]
    print(f"[3a] bge-m3 dense shape={v.shape}")
    free_vram(emb)
    rr = Reranker(C.RERANKER)
    s = rr.score([["What is malaria?", "Malaria is a disease."],
                  ["What is malaria?", "Bananas are yellow."]])
    print(f"[3b] reranker scores={s}  (first should be > second)")
    free_vram(rr)


def check_generation(model_key):
    model_id = {"qwen": C.MODEL_QWEN, "afriqwen": C.MODEL_AFRIQWEN,
                "sunflower": C.MODEL_SUNFLOWER}[model_key]
    subset = {"qwen": "Eng_Uga", "afriqwen": "Swa_Ken", "sunflower": "Lug_Uga"}[model_key]
    msgs = build_messages("What is malaria?",
                          [("What causes malaria?", "Malaria is caused by Plasmodium parasites.")],
                          subset)
    print(f"[4] loading {model_id} via backend={C.GEN_BACKEND} ...")
    if C.GEN_BACKEND == "vllm":
        from vllm import LLM, SamplingParams
        llm = LLM(model=model_id, dtype=C.DTYPE, gpu_memory_utilization=C.GPU_MEM_UTIL,
                  trust_remote_code=True)
        out = llm.chat([msgs], SamplingParams(temperature=0.0, seed=C.SEED, max_tokens=64))
        print(f"[4] OUTPUT: {out[0].outputs[0].text.strip()!r}")
        free_vram(llm)
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        gen = model.generate(**enc, do_sample=False, max_new_tokens=64,
                             pad_token_id=tok.eos_token_id)
        print(f"[4] OUTPUT: {tok.decode(gen[0][enc.input_ids.shape[1]:], skip_special_tokens=True)!r}")
        free_vram(model, tok)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="afriqwen", choices=["qwen", "afriqwen", "sunflower"])
    ap.add_argument("--skip-gen", action="store_true", help="only test torch/data/retrieval")
    args = ap.parse_args()

    check_torch()
    check_data()
    check_retrieval()
    if not args.skip_gen:
        check_generation(args.model)
    print("\nSMOKE TEST PASSED.")
