import argparse
import csv
import importlib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "classification_results"
    / "final_relevant_experiment_summary"
)


@dataclass(frozen=True)
class Experiment:
    group: str
    name: str
    module: str


EXPERIMENTS = [
    # Multiclass model evolution.
    Experiment(
        "multiclass",
        "Mask-aware MTA-Swin",
        "integration_ds2net_maskaware_mtaswin",
    ),
    Experiment(
        "multiclass",
        "Mask-aware + augmentation",
        "integration_ds2net_maskaware_mtaswin_aug",
    ),
    Experiment(
        "multiclass",
        "Mask-aware + augmentation + balanced sampling",
        "integration_ds2net_maskaware_mtaswin_aug_balanced",
    ),
    Experiment(
        "multiclass",
        "Mask-aware + augmentation + focal loss",
        "integration_ds2net_maskaware_mtaswin_aug_focal",
    ),
    # Main binary models.
    Experiment(
        "binary",
        "Binary ROI + mask + augmentation + focal loss",
        "integration_ds2net_maskaware_mtaswin_aug_focal_binary",
    ),
    Experiment(
        "binary",
        "Binary + malignant-only augmentation",
        "integration_ds2net_maskaware_mtaswin_aug_focal_binary_malignant_aug",
    ),
    # Full-image context development.
    Experiment(
        "context",
        "Context 384, batch 1, ResNet18",
        "integration_ds2net_maskaware_mtaswin_binary_context384",
    ),
    Experiment(
        "context",
        "Context 224, batch 1, ResNet18",
        "integration_ds2net_maskaware_mtaswin_binary_context224",
    ),
    Experiment(
        "context",
        "Context 224, batch 2, accumulation 4",
        "integration_ds2net_maskaware_mtaswin_binary_context224_batch2",
    ),
    Experiment(
        "context",
        "Context 224, batch 2, accumulation 2",
        "integration_ds2net_maskaware_mtaswin_binary_context224_batch2_accum2",
    ),
    Experiment(
        "context",
        "Context 224, lightweight CNN + BatchNorm",
        "integration_ds2net_maskaware_mtaswin_binary_context224_lightcnn",
    ),
    Experiment(
        "context",
        "Context 224, lightweight CNN + GroupNorm",
        "integration_ds2net_maskaware_mtaswin_binary_context224_lightcnn_groupnorm",
    ),
    # Controlled ablations.
    Experiment(
        "ablation",
        "A1: Context model without explicit mask",
        "ablation_ds2net_mtaswin_binary_context224_no_mask",
    ),
    Experiment(
        "ablation",
        "Reference: ROI + mask + augmentation + focal",
        "ablation_ds2net_mtaswin_binary_roi_mask_no_context",
    ),
    Experiment(
        "ablation",
        "A3: Without augmentation",
        "ablation_ds2net_mtaswin_binary_roi_mask_no_augmentation",
    ),
    Experiment(
        "ablation",
        "A4: Cross-entropy instead of focal loss",
        "ablation_ds2net_mtaswin_binary_roi_mask_cross_entropy",
    ),
    Experiment(
        "ablation",
        "A5: RGB ROI only",
        "ablation_ds2net_mtaswin_binary_roi_only",
    ),
    Experiment(
        "ablation",
        "A6: Bilinear mask resizing",
        "ablation_ds2net_mtaswin_binary_roi_mask_bilinear_resize",
    ),
    Experiment(
        "ablation",
        "A7: Focal loss without class alpha",
        "ablation_ds2net_mtaswin_binary_roi_mask_focal_no_alpha",
    ),
]


PREFERRED_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "malignant_precision",
    "sensitivity",
    "specificity",
    "malignant_f1",
    "negative_predictive_value",
    "roc_auc",
    "pr_auc",
    "true_negatives",
    "false_positives",
    "false_negatives",
    "true_positives",
]


ALIASES = {
    "balanced_acc": "balanced_accuracy",
    "balancedaccuracy": "balanced_accuracy",
    "precision_malignant": "malignant_precision",
    "malignant_recall": "sensitivity",
    "recall_malignant": "sensitivity",
    "recall": "sensitivity",
    "sensitivity_recall": "sensitivity",
    "f1_malignant": "malignant_f1",
    "negative_predictive_val": "negative_predictive_value",
    "npv": "negative_predictive_value",
    "rocauc": "roc_auc",
    "roc_auc_score": "roc_auc",
    "prauc": "pr_auc",
    "average_precision": "pr_auc",
    "tn": "true_negatives",
    "fp": "false_positives",
    "fn": "false_negatives",
    "tp": "true_positives",
}


def normalize_key(value):
    key = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return ALIASES.get(key, key)


def normalize_value(value):
    text = str(value).strip()
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text

    if number.is_integer() and abs(number) >= 1:
        return str(int(number))
    return f"{number:.4f}"


def read_metric_summary(path):
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))

    if len(rows) < 2:
        raise ValueError("metric summary has no data rows")

    # Standard wide CSV: one header row followed by one result row.
    if len(rows[0]) > 2:
        return {
            normalize_key(key): normalize_value(value)
            for key, value in zip(rows[0], rows[1])
        }

    # Vertical CSV: metric,value pairs.
    output = {}
    start = 1 if normalize_key(rows[0][0]) in {"metric", "name"} else 0
    for row in rows[start:]:
        if len(row) >= 2:
            output[normalize_key(row[0])] = normalize_value(row[1])
    return output


def resolve_summary_path(experiment):
    try:
        config = importlib.import_module(f"{experiment.module}.config")
    except Exception as error:
        return None, f"config import failed: {error}"

    path = getattr(config, "METRIC_SUMMARY_PATH", None)
    if path is None:
        return None, "config has no METRIC_SUMMARY_PATH"
    return Path(path), ""


def run_evaluation(experiment):
    command = [sys.executable, "-m", f"{experiment.module}.evaluate"]
    print("\nRunning:", " ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return completed.returncode == 0


def collect(experiments, evaluate):
    records = []
    skipped = []

    for experiment in experiments:
        if evaluate and not run_evaluation(experiment):
            skipped.append(
                {
                    "group": experiment.group,
                    "experiment": experiment.name,
                    "module": experiment.module,
                    "reason": "evaluation command failed",
                }
            )
            continue

        summary_path, error = resolve_summary_path(experiment)
        if summary_path is None:
            skipped.append(
                {
                    "group": experiment.group,
                    "experiment": experiment.name,
                    "module": experiment.module,
                    "reason": error,
                }
            )
            continue

        if not summary_path.exists():
            skipped.append(
                {
                    "group": experiment.group,
                    "experiment": experiment.name,
                    "module": experiment.module,
                    "reason": f"summary not found: {summary_path}",
                }
            )
            continue

        try:
            metrics = read_metric_summary(summary_path)
        except Exception as error:
            skipped.append(
                {
                    "group": experiment.group,
                    "experiment": experiment.name,
                    "module": experiment.module,
                    "reason": f"could not read summary: {error}",
                }
            )
            continue

        records.append(
            {
                "group": experiment.group,
                "experiment": experiment.name,
                "module": experiment.module,
                "summary_path": str(summary_path),
                **metrics,
            }
        )

    return records, skipped


def fieldnames(records):
    observed = set()
    for record in records:
        observed.update(record)

    fixed = ["group", "experiment", "module", "summary_path"]
    metrics = [key for key in PREFERRED_COLUMNS if key in observed]
    remaining = sorted(observed - set(fixed) - set(metrics) - {"model"})
    return fixed + metrics + remaining


def write_csv(path, records, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def markdown_table(records, columns):
    labels = {
        "accuracy": "Acc.",
        "balanced_accuracy": "Bal. Acc.",
        "macro_f1": "Macro F1",
        "malignant_precision": "Mal. Prec.",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "malignant_f1": "Mal. F1",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
    }
    selected = ["experiment"] + [key for key in columns if key in labels]
    header = "| " + " | ".join(labels.get(key, "Experiment") for key in selected) + " |"
    separator = "| " + " | ".join("---" for _ in selected) + " |"
    lines = [header, separator]
    for record in records:
        lines.append(
            "| "
            + " | ".join(record.get(key, "—") or "—" for key in selected)
            + " |"
        )
    return "\n".join(lines)


def read_multiseed_summary():
    path = (
        PROJECT_ROOT
        / "classification_results"
        / "multiseed_ablation"
        / "mean_std_summary.csv"
    )
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_markdown(records, skipped, multiseed_rows):
    path = OUTPUT_DIR / "final_experiment_summary.md"
    columns = fieldnames(records)
    groups = ["multiclass", "binary", "context", "ablation"]
    titles = {
        "multiclass": "Multiclass model evolution",
        "binary": "Main binary experiments",
        "context": "Full-image context experiments",
        "ablation": "Controlled ablation studies",
    }

    lines = [
        "# Final Relevant Experiment Summary",
        "",
        "Metrics were collected from each experiment's `metric_summary.csv`.",
    ]
    for group in groups:
        group_records = [row for row in records if row["group"] == group]
        if not group_records:
            continue
        lines.extend(
            [
                "",
                f"## {titles[group]}",
                "",
                markdown_table(group_records, columns),
            ]
        )

    if multiseed_rows:
        lines.extend(
            [
                "",
                "## Multi-seed mean ± standard deviation",
                "",
                "| Experiment | Metric | Seeds | Mean ± SD |",
                "| --- | --- | --- | --- |",
            ]
        )
        for row in multiseed_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.get("experiment", "—"),
                        row.get("metric", "—"),
                        row.get("number_of_seeds", "—"),
                        row.get("mean_plus_minus_std", "—"),
                    ]
                )
                + " |"
            )

    if skipped:
        lines.extend(["", "## Skipped experiments", ""])
        for item in skipped:
            lines.append(
                f"- `{item['module']}`: {item['reason']}"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and summarize relevant classification experiments."
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run each evaluate.py before collecting its metric summary.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["multiclass", "binary", "context", "ablation"],
        default=["multiclass", "binary", "context", "ablation"],
    )
    args = parser.parse_args()

    selected = [item for item in EXPERIMENTS if item.group in args.groups]
    records, skipped = collect(selected, args.evaluate)
    multiseed_rows = read_multiseed_summary()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if records:
        columns = fieldnames(records)
        write_csv(OUTPUT_DIR / "all_relevant_experiments.csv", records, columns)
        for group in args.groups:
            group_records = [row for row in records if row["group"] == group]
            if group_records:
                write_csv(
                    OUTPUT_DIR / f"{group}_experiments.csv",
                    group_records,
                    columns,
                )

    if skipped:
        write_csv(
            OUTPUT_DIR / "skipped_experiments.csv",
            skipped,
            ["group", "experiment", "module", "reason"],
        )

    if multiseed_rows:
        write_csv(
            OUTPUT_DIR / "multiseed_mean_std.csv",
            multiseed_rows,
            list(multiseed_rows[0].keys()),
        )

    markdown_path = write_markdown(records, skipped, multiseed_rows)
    print("\nCollected experiments:", len(records))
    print("Skipped experiments:", len(skipped))
    print("Summary directory:", OUTPUT_DIR)
    print("Markdown summary:", markdown_path)


if __name__ == "__main__":
    main()
