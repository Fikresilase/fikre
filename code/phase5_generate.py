"""Phase 5 — generation, grouped by model with FULL teardown between passes.

Pass A: Qwen3.5-35B  -> 4 English subsets  -> wipe
Pass B: AfriqueQwen  -> Aka/Swa/Amh        -> wipe
Pass C: Sunflower    -> Lug                -> wipe

Greedy/deterministic. Checkpoints answers every CHECKPOINT_EVERY rows so a crash resumes.
Backend = vLLM (default) or HF transformers fallback (GEN_BACKEND=hf).
"""
import os

import pandas as pd

import config as C
from prompts import build_messages
from utils import free_vram, load_json, set_seed


# --------------------------------------------------------------------------- helpers
def _load_done():
    """Already-generated IDs (for resume)."""
    if C.ANSWERS_CSV.exists():
        df = pd.read_csv(C.ANSWERS_CSV)
        return df, set(df[C.COL_ID].astype(str))
    return pd.DataFrame(columns=[C.COL_ID, "generated_answer"]), set()


def _append(rows):
    df = pd.DataFrame(rows)
    header = not C.ANSWERS_CSV.exists()
    df.to_csv(C.ANSWERS_CSV, mode="a", header=header, index=False)


def _rows_for_subsets(sample, exemplars, train_q, train_a, subsets, done_ids):
    """Yield (row_id, subset, messages) for not-yet-done rows in the given subsets."""
    sub = sample[sample[C.COL_SUBSET].isin(subsets)]
    for _, r in sub.iterrows():
        rid = str(r[C.COL_ID])
        if rid in done_ids:
            continue
        ex_ids = exemplars.get(rid, [])
        pairs = [(train_q.get(e, ""), train_a.get(e, "")) for e in ex_ids]
        yield rid, r[C.COL_SUBSET], build_messages(str(r[C.COL_IN]), pairs, r[C.COL_SUBSET])


# --------------------------------------------------------------------------- backends
def _gen_vllm(model_id, items):
    """items: list of (row_id, subset, messages). Returns list of (row_id, text)."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_id, dtype=C.DTYPE, gpu_memory_utilization=C.GPU_MEM_UTIL,
              trust_remote_code=True)
    out = []
    buf = []
    # group by subset so max_new_tokens (a SamplingParams field) is consistent per call
    by_subset = {}
    for rid, subset, msgs in items:
        by_subset.setdefault(subset, []).append((rid, msgs))

    for subset, rows in by_subset.items():
        sp = SamplingParams(temperature=0.0, seed=C.SEED,
                            max_tokens=C.MAX_NEW_TOKENS[subset])
        msgs_list = [m for _, m in rows]
        results = llm.chat(msgs_list, sp)
        for (rid, _), res in zip(rows, results):
            text = res.outputs[0].text.strip()
            out.append({C.COL_ID: rid, "generated_answer": text})
            buf.append({C.COL_ID: rid, "generated_answer": text})
            if len(buf) >= C.CHECKPOINT_EVERY:
                _append(buf); buf = []
                print(f"[phase5] checkpoint: {len(out)} rows done")
    if buf:
        _append(buf)
    free_vram(llm)
    return out


def _gen_hf(model_id, items):
    """Transformers fallback: batched greedy generation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # decoder-only batched generation requires LEFT padding; keep the question (prompt tail)
    # when truncating by dropping from the LEFT (oldest exemplars first).
    tok.padding_side = "left"
    tok.truncation_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    out, buf = [], []
    BATCH = int(os.environ.get("HF_BATCH", "8"))

    by_subset = {}
    for rid, subset, msgs in items:
        by_subset.setdefault(subset, []).append((rid, msgs))

    for subset, rows in by_subset.items():
        for i in range(0, len(rows), BATCH):
            chunk = rows[i:i + BATCH]
            prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                       for _, m in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=4096).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=False,
                                     max_new_tokens=C.MAX_NEW_TOKENS[subset],
                                     pad_token_id=tok.eos_token_id)
            for (rid, _), seq, inp in zip(chunk, gen, enc["input_ids"]):
                text = tok.decode(seq[inp.shape[0]:], skip_special_tokens=True).strip()
                out.append({C.COL_ID: rid, "generated_answer": text})
                buf.append({C.COL_ID: rid, "generated_answer": text})
            if len(buf) >= C.CHECKPOINT_EVERY:
                _append(buf); buf = []
                print(f"[phase5] checkpoint: {len(out)} rows done")
    if buf:
        _append(buf)
    free_vram(model, tok)
    return out


# --------------------------------------------------------------------------- run
def run():
    set_seed(C.SEED)
    sample = pd.read_csv(C.SAMPLE_CSV)
    train = pd.read_csv(C.TRAIN_CSV)
    train_q = dict(zip(train[C.COL_ID].astype(str), train[C.COL_IN].fillna("").astype(str)))
    train_a = dict(zip(train[C.COL_ID].astype(str), train[C.COL_OUT].fillna("").astype(str)))
    exemplars = load_json(C.EXEMPLARS_JSON)

    _, done_ids = _load_done()
    gen = _gen_vllm if C.GEN_BACKEND == "vllm" else _gen_hf
    print(f"[phase5] backend={C.GEN_BACKEND}; {len(done_ids)} rows already done.")

    for model_id, subsets in C.MODEL_PASSES:
        items = list(_rows_for_subsets(sample, exemplars, train_q, train_a, subsets, done_ids))
        if not items:
            print(f"[phase5] pass {model_id} {subsets}: nothing to do (resumed). Skipping load.")
            continue
        print(f"[phase5] === loading {model_id} for {subsets} ({len(items)} rows) ===")
        gen(model_id, items)
        print(f"[phase5] === freed {model_id} ===")

    print(f"[phase5] answers -> {C.ANSWERS_CSV}")


if __name__ == "__main__":
    run()
