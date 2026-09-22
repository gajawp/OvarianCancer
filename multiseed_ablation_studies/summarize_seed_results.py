import csv
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = (
    PROJECT_ROOT
    / "classification_results"
    / "multiseed_ablation"
)

EXPERIMENTS = [
    "reference",
    "no_augmentation",
    "roi_only",
]

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "malignant_precision",
    "sensitivity",
    "specificity",
    "malignant_f1",
    "macro_f1",
    "negative_predictive_value",
    "roc_auc",
    "pr_auc",
]


def read_runs(experiment):
    runs = []
    experiment_directory = RESULTS_ROOT / experiment

    if not experiment_directory.exists():
        return runs

    for seed_directory in sorted(experiment_directory.glob("seed_*")):
        summary_path = seed_directory / "metric_summary.csv"
        if not summary_path.exists():
            continue

        with summary_path.open("r", encoding="utf-8", newline="") as file:
            row = next(csv.DictReader(file))

        runs.append(
            {
                "experiment": experiment,
                "seed": int(seed_directory.name.removeprefix("seed_")),
                **row,
            }
        )

    return runs


def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    all_runs = []

    for experiment in EXPERIMENTS:
        all_runs.extend(read_runs(experiment))

    if not all_runs:
        print("No completed metric_summary.csv files found.")
        return

    per_seed_path = RESULTS_ROOT / "all_seed_metrics.csv"
    per_seed_fields = [
        "experiment",
        "seed",
        *[key for key in all_runs[0] if key not in {"experiment", "seed"}],
    ]

    with per_seed_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=per_seed_fields)
        writer.writeheader()
        writer.writerows(all_runs)

    aggregate_rows = []
    for experiment in EXPERIMENTS:
        runs = [
            row for row in all_runs
            if row["experiment"] == experiment
        ]
        if not runs:
            continue

        for metric in METRICS:
            values = [float(row[metric]) for row in runs]
            mean = statistics.mean(values)
            sample_std = (
                statistics.stdev(values)
                if len(values) >= 2
                else 0.0
            )
            aggregate_rows.append(
                {
                    "experiment": experiment,
                    "metric": metric,
                    "number_of_seeds": len(values),
                    "mean": mean,
                    "sample_standard_deviation": sample_std,
                    "mean_plus_minus_std": (
                        f"{mean:.4f} ± {sample_std:.4f}"
                    ),
                }
            )

    aggregate_path = RESULTS_ROOT / "mean_std_summary.csv"
    with aggregate_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(aggregate_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print("Saved per-seed metrics:", per_seed_path)
    print("Saved mean ± standard deviation:", aggregate_path)


if __name__ == "__main__":
    main()
