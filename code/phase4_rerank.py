"""Phase 4 — bge-reranker-v2-m3 over (Val question, Train question) -> top-N_EXEMPLARS.

Rerank on question<->question similarity: the most relevant exemplars are Train rows whose
QUESTION matches; we then show their Q&A pair to the generator. Free the reranker after.
"""
import pandas as pd

import config as C
from utils import free_vram, load_json, save_json


def run():
    if C.EXEMPLARS_JSON.exists():
        print(f"[phase4] exemplars cached -> {C.EXEMPLARS_JSON}; skipping.")
        return

    from rerank_model import Reranker

    sample = pd.read_csv(C.SAMPLE_CSV)
    train = pd.read_csv(C.TRAIN_CSV)
    train_q = dict(zip(train[C.COL_ID].astype(str), train[C.COL_IN].fillna("").astype(str)))
    candidates = load_json(C.CANDIDATES_JSON)

    reranker = Reranker(C.RERANKER)
    exemplars = {}
    val_q = dict(zip(sample[C.COL_ID].astype(str), sample[C.COL_IN].fillna("").astype(str)))

    for n, (row_id, cand_ids) in enumerate(candidates.items()):
        q = val_q[row_id]
        pairs = [[q, train_q.get(cid, "")] for cid in cand_ids]
        scores = reranker.score(pairs)
        ranked = [cid for _, cid in sorted(zip(scores, cand_ids), key=lambda x: -x[0])]
        exemplars[row_id] = ranked[:C.N_EXEMPLARS]
        if (n + 1) % 200 == 0:
            print(f"[phase4] reranked {n + 1}/{len(candidates)}")

    save_json(exemplars, C.EXEMPLARS_JSON)
    print(f"[phase4] wrote {len(exemplars)} rows x {C.N_EXEMPLARS} -> {C.EXEMPLARS_JSON}")
    free_vram(reranker)


if __name__ == "__main__":
    run()
