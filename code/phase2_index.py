"""Phase 2 — embed all Train `input` (questions) with bge-m3 dense -> cached index."""
import numpy as np
import pandas as pd

import config as C
from utils import free_vram


def run():
    if C.TRAIN_EMB_NPY.exists() and C.TRAIN_IDS_NPY.exists():
        print(f"[phase2] index cached -> {C.TRAIN_EMB_NPY}; skipping.")
        return

    from FlagEmbedding import BGEM3FlagModel

    train = pd.read_csv(C.TRAIN_CSV)
    texts = train[C.COL_IN].fillna("").astype(str).tolist()
    ids = train[C.COL_ID].astype(str).tolist()

    print(f"[phase2] embedding {len(texts)} Train inputs with {C.EMBEDDER} ...")
    model = BGEM3FlagModel(C.EMBEDDER, use_fp16=True)
    out = model.encode(texts, batch_size=64, max_length=512)
    emb = np.asarray(out["dense_vecs"], dtype=np.float32)
    # L2-normalize so dot product == cosine similarity.
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

    np.save(C.TRAIN_EMB_NPY, emb)
    np.save(C.TRAIN_IDS_NPY, np.array(ids))
    print(f"[phase2] wrote {emb.shape} -> {C.TRAIN_EMB_NPY}")
    free_vram(model)


if __name__ == "__main__":
    run()
