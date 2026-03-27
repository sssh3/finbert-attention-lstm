"""
main.py
-------
Single entry point for the full training pipeline.

Steps
-----
  pipeline  — Download prices, compute features, align news, create splits.
  embed     — Encode headlines with FinBERT + PCA (slow; skip if done).
  train     — Train all models per ticker (optionally with Optuna tuning).

Each step is skipped automatically if its output already exists, unless
--force is passed.

Usage examples
--------------
  # Run everything end-to-end for the first time:
  python main.py --all

  # Full pipeline with Optuna tuning (50 trials, 3 seeds):
  python main.py --all --tune

  # Only training (data + embeddings already on disk):
  python main.py --steps train

  # Training with Optuna, custom trials / seeds:
  python main.py --steps train --tune --trials 50 --seeds 10

  # Force re-run a step even if outputs exist:
  python main.py --steps embed --force

  # Run multiple specific steps:
  python main.py --steps pipeline embed
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# Output sentinels — if these exist the step is considered done.
_OUTPUTS = {
    "pipeline": ROOT / "data" / "processed" / "features.parquet",
    "embed":    ROOT / "data" / "processed" / "embeddings_pca.npy",
    "train":    ROOT / "results"            / "metrics.json",
}  

ALL_STEPS = ["pipeline", "embed", "train"]


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def run_pipeline(force: bool) -> None:
    sentinel = _OUTPUTS["pipeline"]
    if sentinel.exists() and not force:
        print(f"[pipeline] Output already exists ({sentinel}). Skipping. (use --force to re-run)")
        return

    print("\n" + "=" * 60)
    print("STEP 1 — Data pipeline")
    print("=" * 60)
    import data_pipeline
    t0 = time.time()
    data_pipeline.main()
    print(f"[pipeline] Done in {time.time() - t0:.1f}s")


def run_embed(force: bool) -> None:
    sentinel = _OUTPUTS["embed"]
    if sentinel.exists() and not force:
        print(f"[embed] Output already exists ({sentinel}). Skipping. (use --force to re-run)")
        return

    features_file = _OUTPUTS["pipeline"]
    if not features_file.exists():
        sys.exit("[embed] features.parquet not found — run 'pipeline' step first.")

    print("\n" + "=" * 60)
    print("STEP 2 — FinBERT embedding + PCA")
    print("=" * 60)
    import embed_news
    t0 = time.time()
    embed_news.main()
    print(f"[embed] Done in {time.time() - t0:.1f}s")


def run_train(force: bool, tune: bool, trials: int, seeds: int) -> None:
    sentinel = _OUTPUTS["train"]
    if sentinel.exists() and not force:
        print(f"[train] Output already exists ({sentinel}). Skipping. (use --force to re-run)")
        return

    for dep_step in ["pipeline", "embed"]:
        if not _OUTPUTS[dep_step].exists():
            sys.exit(f"[train] '{dep_step}' output not found — run that step first.")

    print("\n" + "=" * 60)
    print("STEP 3 — Model training" + (" + Optuna tuning" if tune else ""))
    print("=" * 60)

    # Inject CLI args into train.py's argument parser
    import train
    sys.argv = ["train.py"]
    if tune:
        sys.argv += ["--tune", "--trials", str(trials)]
    sys.argv += ["--seeds", str(seeds)]

    t0 = time.time()
    train.main()
    print(f"[train] Done in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Which steps to run
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all", action="store_true",
        help="Run all steps in order: pipeline → embed → train",
    )
    group.add_argument(
        "--steps", nargs="+", choices=ALL_STEPS, metavar="STEP",
        help=f"Steps to run (choices: {', '.join(ALL_STEPS)})",
    )

    # General flags
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run a step even if its output already exists",
    )

    # Train-specific flags
    parser.add_argument(
        "--tune", action="store_true",
        help="(train step) Run Optuna hyperparameter search",
    )
    parser.add_argument(
        "--trials", type=int, default=30,
        help="(train step) Number of Optuna trials per model (default: 30)",
    )
    parser.add_argument(
        "--seeds", type=int, default=3,
        help="(train step) Random seeds for final training (default: 3)",
    )

    args = parser.parse_args()
    steps = ALL_STEPS if args.all else args.steps

    total_t0 = time.time()
    for step in steps:
        if step == "pipeline":
            run_pipeline(args.force)
        elif step == "embed":
            run_embed(args.force)
        elif step == "train":
            run_train(args.force, args.tune, args.trials, args.seeds)

    elapsed = time.time() - total_t0
    print(f"\nAll done. Total time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
