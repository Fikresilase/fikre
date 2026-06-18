"""Run Phases 1-6 in order. Each phase is crash-resumable from cached artifacts."""
import config as C  # noqa: F401  (ensures ARTIFACTS dir + config load first)
import phase1_sample
import phase2_index
import phase3_retrieve
import phase4_rerank
import phase5_generate
import phase6_score


def main():
    print("========== PHASE 1: sample ==========");    phase1_sample.run()
    print("========== PHASE 2: index ===========");    phase2_index.run()
    print("========== PHASE 3: retrieve ========");    phase3_retrieve.run()
    print("========== PHASE 4: rerank ==========");    phase4_rerank.run()
    print("========== PHASE 5: generate ========");    phase5_generate.run()
    print("========== PHASE 6: score ===========");    phase6_score.run()


if __name__ == "__main__":
    main()
