"""Phase 3 — embed Val sample questions, dense search Train -> top-N_CANDIDATES per row."""
import numpy as np
import pandas as pd

import config as C
from utils import free_vram, save_json


def run():
    if C.CANDIDATES_JSON.exists():
        print(f"[phase3] candidates cached -> {C.CANDIDATES_JSON}; skipping.")
        return

    from FlagEmbedding import BGEM3FlagModel

    sample = pd.read_csv(C.SAMPLE_CSV)
    train_emb = np.load(C.TRAIN_EMB_NPY)            # (n_train, d), normalized
    train_ids = np.load(C.TRAIN_IDS_NPY).astype(str)

    q_texts = sample[C.COL_IN].fillna("").astype(str).tolist()
    print(f"[phase3] embedding {len(q_texts)} Val questions ...")
    model = BGEM3FlagModel(C.EMBEDDER, use_fp16=True)
    q = np.asarray(model.encode(q_texts, batch_size=64, max_length=512)["dense_vecs"],
                   dtype=np.float32)
    q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    free_vram(model)

    # cosine sim via matmul; top-N per row.
    sims = q @ train_emb.T                            # (n_val, n_train)
    topk = np.argpartition(-sims, C.N_CANDIDATES, axis=1)[:, :C.N_CANDIDATES]
    candidates = {}
    for i, row_id in enumerate(sample[C.COL_ID].astype(str)):
        idx = topk[i]
        idx = idx[np.argsort(-sims[i, idx])]          # sort the top-N by score
        candidates[row_id] = [train_ids[j] for j in idx]

    save_json(candidates, C.CANDIDATES_JSON)
    print(f"[phase3] wrote {len(candidates)} rows x {C.N_CANDIDATES} cands -> {C.CANDIDATES_JSON}")


if __name__ == "__main__":
    run()
