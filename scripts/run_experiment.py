"""Phase 18: reproducible experiment runner.

Usage:
  python scripts/run_experiment.py --quick        # reuse preprocessing/index artifacts
  python scripts/run_experiment.py                # full pipeline (rebuilds artifacts if missing)
  python scripts/run_experiment.py --from-scratch # force rebuild raw->processed->index

Every step prints a runtime; a consolidated results/eval_summary.json is written
at the end. --quick target: < 15 min on this machine.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = dict(os.environ)
ENV.setdefault("PYTHONIOENCODING", "utf-8")


def step(name: str, script: str, args: list[str] | None = None,
         skip_if: Path | None = None, force: bool = False) -> float:
    if skip_if and skip_if.exists() and not force:
        print(f"[skip] {name} (artifact exists: {skip_if.name})")
        return 0.0
    print(f"[run ] {name} ... ", flush=True)
    t0 = time.time()
    cmd = [sys.executable, str(ROOT / script)] + (args or [])
    proc = subprocess.run(cmd, cwd=ROOT, env=ENV)
    if proc.returncode != 0:
        raise SystemExit(f"[FAIL] {name} exited {proc.returncode}")
    dt = round(time.time() - t0, 1)
    print(f"[done] {name} ({dt}s)\n")
    return dt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="skip preprocessing/index rebuild")
    p.add_argument("--from-scratch", action="store_true", help="force full rebuild")
    args = p.parse_args()

    force = args.from_scratch
    times = {}
    (ROOT / "results").mkdir(exist_ok=True)

    times["preprocessing"] = step(
        "preprocessing (raw -> pairs)", "scripts/preprocess.py",
        skip_if=ROOT / "data/processed/spotify_pairs.csv", force=force,
    )
    times["golden_build"] = step(
        "golden set build", "scripts/build_golden_set.py",
        skip_if=ROOT / "data/golden/golden_candidates.csv", force=force,
    )
    times["majority_baseline"] = step("majority baseline", "scripts/majority_baseline.py")
    times["rule_baseline"] = step("rule baseline", "scripts/rule_baseline.py")
    times["classifier_baseline"] = step("tfidf classifier baseline", "scripts/classifier_baseline.py")
    times["llm_intent_eval"] = step("llm intent classifier eval", "scripts/llm_intent_eval.py")
    times["retrieval_index"] = step(
        "retrieval index build", "scripts/../src/retrieval.py",
        skip_if=ROOT / "data/processed/retrieval_index/retrieval_index.joblib", force=force,
    )
    times["retrieval_knn"] = step("retrieval kNN eval", "scripts/retrieval_knn_eval.py")
    times["pipeline_eval"] = step("pipeline eval", "scripts/pipeline_eval.py")
    times["judge_eval"] = step("judge eval", "scripts/judge_eval.py")
    times["failure_analysis"] = step("failure analysis", "scripts/failure_analysis.py")
    times["misleading_metrics"] = step("misleading metrics", "scripts/misleading_metrics.py")

    summary = {
        "mode": "quick" if args.quick else ("from_scratch" if force else "full"),
        "step_runtimes_seconds": times,
        "total_seconds": round(sum(times.values()), 1),
    }
    for fname, keys in [
        ("majority_results.json", ["accuracy", "macro_f1", "weighted_f1"]),
        ("rule_results.json", ["accuracy", "macro_f1", "weighted_f1"]),
        ("classifier_results.json", ["accuracy", "macro_f1", "weighted_f1"]),
        ("llm_classifier_results.json", ["accuracy", "macro_f1", "weighted_f1"]),
        ("retrieval_knn_results.json", ["accuracy", "macro_f1", "weighted_f1"]),
        ("pipeline_results.json", ["intent_accuracy", "intent_macro_f1"]),
    ]:
        f = ROOT / "results" / fname
        if f.exists():
            data = json.loads(f.read_text(encoding="utf-8"))
            summary[fname.replace(".json", "")] = {k: data[k] for k in keys if k in data}
    (ROOT / "results" / "eval_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\nAll steps finished in {summary['total_seconds']}s")
    for fname in ["majority", "rule", "classifier", "llm_classifier",
                  "retrieval_knn", "pipeline"]:
        d = summary.get(fname)
        if d:
            print(f"  {fname:>16}: " + "  ".join(f"{k}={v:.3f}" for k, v in d.items()))


if __name__ == "__main__":
    main()