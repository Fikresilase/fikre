"""Phase 1 — draw a Val subset whose per-subset counts EXACTLY match the Test set.

We optimize for Test, so the Val proxy is sampled to mirror Test's subset distribution.
If Val lacks enough rows in a subset, take all available and log the shortfall.
"""
import pandas as pd

import config as C
from utils import set_seed


def run():
    set_seed(C.SEED)
    test = pd.read_csv(C.TEST_CSV)
    val = pd.read_csv(C.VAL_CSV)

    test_counts = test[C.COL_SUBSET].value_counts().to_dict()
    print(f"[phase1] Test per-subset counts: {test_counts}")

    parts = []
    for subset, n in test_counts.items():
        pool = val[val[C.COL_SUBSET] == subset]
        if len(pool) < n:
            print(f"[phase1] WARNING: Val {subset} has {len(pool)} < {n} requested; taking all.")
            take = pool
        else:
            take = pool.sample(n=n, random_state=C.SEED)
        parts.append(take)

    sample = pd.concat(parts).reset_index(drop=True)
    sample.to_csv(C.SAMPLE_CSV, index=False)
    print(f"[phase1] wrote {len(sample)} rows -> {C.SAMPLE_CSV}")
    print(sample[C.COL_SUBSET].value_counts())
    return sample


if __name__ == "__main__":
    run()
