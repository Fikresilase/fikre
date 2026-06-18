"""Stage G — generate predictions on the REAL Test set and write the Zindi submission.

Reuses the same retrieval+rerank+generate logic but over Test (no `output` column).
Submission format: ID, TargetRLF1, TargetR1F1, TargetLLM  (the 3 target cols identical).

Run after the Val harness looks good:
  python /workspace/code/predict_test.py
"""
import numpy as np
import pandas as pd

import config as C
from prompts import build_messages
from utils import free_vram, set_seed

TEST_CAND = C.ARTIFACTS / "test_candidates.json"
TEST_EXEM = C.ARTIFACTS / "test_exemplars.json"
TEST_ANS = C.ARTIFACTS / "test_answers.csv"
SUBMISSION = C.ARTIFACTS / "submission.csv"


def retrieve_and_rerank():
    from FlagEmbedding import BGEM3FlagModel, FlagReranker
    from utils import save_json

    test = pd.read_csv(C.TEST_CSV)
    train = pd.read_csv(C.TRAIN_CSV)
    train_emb = np.load(C.TRAIN_EMB_NPY)
    train_ids = np.load(C.TRAIN_IDS_NPY).astype(str)
    train_q = dict(zip(train[C.COL_ID].astype(str), train[C.COL_IN].fillna("").astype(str)))

    emb = BGEM3FlagModel(C.EMBEDDER, use_fp16=True)
    q = np.asarray(emb.encode(test[C.COL_IN].fillna("").astype(str).tolist(),
                              batch_size=64, max_length=512)["dense_vecs"], dtype=np.float32)
    q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    free_vram(emb)

    sims = q @ train_emb.T
    topk = np.argpartition(-sims, C.N_CANDIDATES, axis=1)[:, :C.N_CANDIDATES]
    cands = {}
    for i, rid in enumerate(test[C.COL_ID].astype(str)):
        idx = topk[i][np.argsort(-sims[i, topk[i]])]
        cands[rid] = [train_ids[j] for j in idx]
    save_json(cands, TEST_CAND)

    rr = FlagReranker(C.RERANKER, use_fp16=True)
    val_q = dict(zip(test[C.COL_ID].astype(str), test[C.COL_IN].fillna("").astype(str)))
    exem = {}
    for rid, cand_ids in cands.items():
        pairs = [[val_q[rid], train_q.get(c, "")] for c in cand_ids]
        scores = rr.compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = [scores]
        ranked = [c for _, c in sorted(zip(scores, cand_ids), key=lambda x: -x[0])]
        exem[rid] = ranked[:C.N_EXEMPLARS]
    save_json(exem, TEST_EXEM)
    return test, exem


def generate(test, exemplars):
    from utils import load_json
    train = pd.read_csv(C.TRAIN_CSV)
    train_q = dict(zip(train[C.COL_ID].astype(str), train[C.COL_IN].fillna("").astype(str)))
    train_a = dict(zip(train[C.COL_ID].astype(str), train[C.COL_OUT].fillna("").astype(str)))

    done = set()
    if TEST_ANS.exists():
        done = set(pd.read_csv(TEST_ANS)[C.COL_ID].astype(str))

    from vllm import LLM, SamplingParams

    for model_id, subsets in C.MODEL_PASSES:
        sub = test[test[C.COL_SUBSET].isin(subsets)]
        rows = [(str(r[C.COL_ID]), r[C.COL_SUBSET], str(r[C.COL_IN]))
                for _, r in sub.iterrows() if str(r[C.COL_ID]) not in done]
        if not rows:
            continue
        print(f"[predict] loading {model_id} for {subsets} ({len(rows)} rows)")
        llm = LLM(model=model_id, dtype=C.DTYPE, gpu_memory_utilization=C.GPU_MEM_UTIL,
                  trust_remote_code=True)
        by_subset = {}
        for rid, subset, qtext in rows:
            by_subset.setdefault(subset, []).append((rid, qtext))
        for subset, items in by_subset.items():
            sp = SamplingParams(temperature=0.0, seed=C.SEED, max_tokens=C.MAX_NEW_TOKENS[subset])
            msgs = [build_messages(qt, [(train_q.get(e, ""), train_a.get(e, ""))
                                        for e in exemplars[rid]], subset)
                    for rid, qt in items]
            res = llm.chat(msgs, sp)
            out = [{C.COL_ID: rid, "generated_answer": r.outputs[0].text.strip()}
                   for (rid, _), r in zip(items, res)]
            pd.DataFrame(out).to_csv(TEST_ANS, mode="a", header=not TEST_ANS.exists(), index=False)
        free_vram(llm)


def build_submission():
    test = pd.read_csv(C.TEST_CSV)[[C.COL_ID]].astype({C.COL_ID: str})
    ans = pd.read_csv(TEST_ANS).astype({C.COL_ID: str}).drop_duplicates(C.COL_ID)
    # merge onto the full Test ID list (original order) so NO row is dropped
    merged = test.merge(ans, on=C.COL_ID, how="left")
    missing = merged["generated_answer"].isna().sum()
    if missing:
        print(f"[predict] WARNING: {missing} Test rows have no answer; filling blank.")
        merged["generated_answer"] = merged["generated_answer"].fillna("")
    # Zindi multi-metric: the predicted ANSWER text goes in all 3 target cols (identical).
    sub = pd.DataFrame({
        C.COL_ID: merged[C.COL_ID],
        "TargetRLF1": merged["generated_answer"],
        "TargetR1F1": merged["generated_answer"],
        "TargetLLM": merged["generated_answer"],
    })
    sub.to_csv(SUBMISSION, index=False)
    print(f"[predict] submission ({len(sub)} rows, {missing} blank) -> {SUBMISSION}")


def run():
    set_seed(C.SEED)
    test, exemplars = retrieve_and_rerank()
    generate(test, exemplars)
    build_submission()


if __name__ == "__main__":
    run()
