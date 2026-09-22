import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = {
    "reference": "multiseed_ablation_studies.seedstudy_reference",
    "no_augmentation": (
        "multiseed_ablation_studies.seedstudy_no_augmentation"
    ),
    "roi_only": "multiseed_ablation_studies.seedstudy_roi_only",
}

LEGACY_SEED42_RESULTS = {
    "reference": (
        PROJECT_ROOT
        / "classification_results"
        / "ablation_ds2net_mtaswin_binary_roi_mask_no_context"
        / "metric_summary.csv"
    ),
    "no_augmentation": (
        PROJECT_ROOT
        / "classification_results"
        / "ablation_ds2net_mtaswin_binary_roi_mask_no_augmentation"
        / "metric_summary.csv"
    ),
    "roi_only": (
        PROJECT_ROOT
        / "classification_results"
        / "ablation_ds2net_mtaswin_binary_roi_only"
        / "metric_summary.csv"
    ),
}


def result_path(experiment, seed):
    return (
        PROJECT_ROOT
        / "classification_results"
        / "multiseed_ablation"
        / experiment
        / f"seed_{seed}"
        / "metric_summary.csv"
    )


def reuse_existing_seed42(experiment):
    source = LEGACY_SEED42_RESULTS[experiment]
    destination = result_path(experiment, 42)

    if destination.exists():
        return True
    if not source.exists():
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"Reused existing seed-42 metrics: {source}")
    return True


def run_module(module, seed):
    environment = os.environ.copy()
    environment["EXPERIMENT_SEED"] = str(seed)

    command = [sys.executable, "-m", module]
    print("Running:", " ".join(command), f"(seed={seed})")
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate selected ablations across random seeds."
        )
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=sorted(EXPERIMENTS),
        default=list(EXPERIMENTS),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 2026],
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun a seed even when metric_summary.csv already exists.",
    )
    parser.add_argument(
        "--rerun-seed42",
        action="store_true",
        help="Do not reuse metric_summary.csv from the completed seed-42 run.",
    )
    args = parser.parse_args()

    for experiment in args.experiments:
        package = EXPERIMENTS[experiment]

        for seed in args.seeds:
            output = result_path(experiment, seed)

            if output.exists() and not args.force:
                print(f"Skipping completed run: {experiment}, seed {seed}")
                continue

            if (
                seed == 42
                and not args.force
                and not args.rerun_seed42
                and reuse_existing_seed42(experiment)
            ):
                continue

            run_module(f"{package}.train", seed)
            run_module(f"{package}.evaluate", seed)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "multiseed_ablation_studies.summarize_seed_results",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
