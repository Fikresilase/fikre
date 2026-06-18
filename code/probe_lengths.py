"""Optional probe — measure real prompt token lengths per subset to size concurrency.

Run AFTER phase4 (needs exemplars.json). Tokenizes the actual 8-exemplar prompts with each
subset's own model tokenizer and reports p50/p95/max tokens per subset. The high-token subsets
(usually Amharic/Luganda) decide how wide vLLM can batch.

  python /workspace/code/probe_lengths.py
"""
import numpy as np
import pandas as pd

import config as C
from prompts import build_messages
from utils import load_json


def run():
    from transformers import AutoTokenizer

    sample = pd.read_csv(C.SAMPLE_CSV)
    train = pd.read_csv(C.TRAIN_CSV)
    train_q = dict(zip(train[C.COL_ID].astype(str), train[C.COL_IN].fillna("").astype(str)))
    train_a = dict(zip(train[C.COL_ID].astype(str), train[C.COL_OUT].fillna("").astype(str)))
    exemplars = load_json(C.EXEMPLARS_JSON)

    tok_cache = {}
    print(f"{'subset':10} {'n':>5} {'p50':>6} {'p95':>6} {'max':>6}  model")
    for subset, model_id in C.SUBSET_MODEL.items():
        if model_id not in tok_cache:
            tok_cache[model_id] = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tok = tok_cache[model_id]
        rows = sample[sample[C.COL_SUBSET] == subset]
        lens = []
        for _, r in rows.iterrows():
            rid = str(r[C.COL_ID])
            pairs = [(train_q.get(e, ""), train_a.get(e, "")) for e in exemplars.get(rid, [])]
            msgs = build_messages(str(r[C.COL_IN]), pairs, subset)
            try:
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                text = msgs[0]["content"] + "\n" + msgs[1]["content"]
            lens.append(len(tok(text).input_ids))
        if lens:
            a = np.array(lens)
            print(f"{subset:10} {len(a):>5} {int(np.percentile(a,50)):>6} "
                  f"{int(np.percentile(a,95)):>6} {int(a.max()):>6}  {model_id}")

    print("\nKV-cache rule of thumb (Qwen-14B class, bf16): ~0.19 MB/token/seq.")
    print("max_concurrent ~= (free_GB*1024) / (0.19 * p95_tokens). Set max_num_seqs below that.")


if __name__ == "__main__":
    run()
