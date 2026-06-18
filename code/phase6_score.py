"""Phase 6 — ROUGE-1/L F1 per row, per subset, overall (+ 0.37-weighted lexical)."""
import pandas as pd

import config as C


def run():
    from rouge_score import rouge_scorer

    sample = pd.read_csv(C.SAMPLE_CSV)
    answers = pd.read_csv(C.ANSWERS_CSV)
    df = sample.merge(answers, on=C.COL_ID, how="inner")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rows = []
    for _, r in df.iterrows():
        ref = str(r[C.COL_OUT]) if pd.notna(r[C.COL_OUT]) else ""
        gen = str(r["generated_answer"]) if pd.notna(r["generated_answer"]) else ""
        s = scorer.score(ref, gen)
        r1, rl = s["rouge1"].fmeasure, s["rougeL"].fmeasure
        rows.append({
            "subset": r[C.COL_SUBSET],
            "question": r[C.COL_IN],
            "generated_answer": gen,
            "reference_answer": ref,
            "gen_len": len(gen.split()),
            "ref_len": len(ref.split()),
            "rouge1_f1": r1,
            "rougeL_f1": rl,
            "w_r1": 0.37 * r1,
            "w_rl": 0.37 * rl,
        })

    table = pd.DataFrame(rows)
    table.to_csv(C.SCORES_TABLE_CSV, index=False)

    per_subset = table.groupby("subset").agg(
        n=("rouge1_f1", "size"),
        rouge1_f1=("rouge1_f1", "mean"),
        rougeL_f1=("rougeL_f1", "mean"),
        w_r1=("w_r1", "mean"),
        w_rl=("w_rl", "mean"),
    ).round(4)

    overall_r1 = table["rouge1_f1"].mean()
    overall_rl = table["rougeL_f1"].mean()
    combined_lexical = (table["w_r1"] + table["w_rl"]).mean()

    print("\n================ PER-SUBSET ================")
    print(per_subset.to_string())
    print("\n================ OVERALL ===================")
    print(f"ROUGE-1 F1        : {overall_r1:.4f}")
    print(f"ROUGE-L F1        : {overall_rl:.4f}")
    print(f"combined lexical  : {combined_lexical:.4f}  (mean 0.37*R1 + 0.37*RL, = 0.74 of LB)")
    print(f"\n[phase6] table -> {C.SCORES_TABLE_CSV}")
    return table


if __name__ == "__main__":
    run()
