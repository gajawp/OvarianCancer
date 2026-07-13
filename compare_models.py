"""
compare_models.py

Place this file in the root OVARIAN_CANCER directory.

Expected structure:

OVARIAN_CANCER/
├── compare_models.py
├── attention_unet/
├── deeplabv3plus/
├── ds2net/
├── dsnet/
├── segformer/
├── transunet/
├── common/
├── datasets/
└── results/

This script:
1. Runs the existing evaluation script inside every model folder.
2. Extracts Loss, Dice, IoU, Precision, Recall, Specificity,
   and Hausdorff Distance from the printed evaluation output.
3. Creates a CSV automatically.
4. Creates comparison graphs.
5. Does not retrain models.
6. Does not modify existing project files.
"""

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Project configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_FOLDERS = {
    "Attention U-Net": "attention_unet",
    "DeepLabV3+": "deeplabv3plus",
    "DS2Net": "ds2net",
    "DSNet": "dsnet",
    "SegFormer": "segformer",
    "TransUNet": "transunet",
    "U-Net++": "unetplusplus"
}

EVALUATION_SCRIPT_NAMES = [
    "evaluate.py",
    "evaluation.py",
    "eval.py",
    "test.py",
]

OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "model_comparison"
CSV_OUTPUT_PATH = OUTPUT_DIRECTORY / "model_comparison_results.csv"

MAIN_METRICS = [
    "Dice",
    "IoU",
    "Precision",
    "Recall",
    "Specificity",
]

LOWER_IS_BETTER_METRICS = [
    "Loss",
    "Hausdorff Distance",
]


# ============================================================
# Metric extraction patterns
# ============================================================

NUMBER_PATTERN = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"

METRIC_PATTERNS = {
    "Loss": [
        rf"\btest\s+loss\s*[:=]\s*{NUMBER_PATTERN}",
        rf"\bvalidation\s+loss\s*[:=]\s*{NUMBER_PATTERN}",
        rf"\bval\s+loss\s*[:=]\s*{NUMBER_PATTERN}",
        rf"\baverage\s+loss\s*[:=]\s*{NUMBER_PATTERN}",
        rf"\bavg\s+loss\s*[:=]\s*{NUMBER_PATTERN}",
        rf"\bloss\s*[:=]\s*{NUMBER_PATTERN}",
    ],

    "Dice": [
        rf"\bdice\s+score\s*[:=]\s*{NUMBER_PATTERN}",
        rf"\bmean\s+dice\s*[:=]\s*{NUMBER_PATTERN}",
        rf"\bdice\s*[:=]\s*{NUMBER_PATTERN}",
        rf"\bdsc\s*[:=]\s*{NUMBER_PATTERN}",
    ],

    "IoU": [
        rf"\bmean\s+iou\s*[:=]\s*{NUMBER_PATTERN}",
        rf"\biou\s+score\s*[:=]\s*{NUMBER_PATTERN}",
        rf"\bmiou\s*[:=]\s*{NUMBER_PATTERN}",
        rf"\biou\s*[:=]\s*{NUMBER_PATTERN}",
        rf"\bjaccard\s+score\s*[:=]\s*{NUMBER_PATTERN}",
        rf"\bjaccard\s*[:=]\s*{NUMBER_PATTERN}",
    ],

    "Precision": [
        rf"\bprecision\s+score\s*[:=]\s*{NUMBER_PATTERN}",
        rf"\bprecision\s*[:=]\s*{NUMBER_PATTERN}",
    ],

    "Recall": [
        rf"\brecall\s+score\s*[:=]\s*{NUMBER_PATTERN}",
        rf"\brecall\s*[:=]\s*{NUMBER_PATTERN}",
        rf"\bsensitivity\s*[:=]\s*{NUMBER_PATTERN}",
        rf"\btpr\s*[:=]\s*{NUMBER_PATTERN}",
    ],

    "Specificity": [
        rf"\bspecificity\s+score\s*[:=]\s*{NUMBER_PATTERN}",
        rf"\bspecificity\s*[:=]\s*{NUMBER_PATTERN}",
        rf"\btnr\s*[:=]\s*{NUMBER_PATTERN}",
    ],

    "Hausdorff Distance": [
        rf"\bhausdorff\s+distance\s*[:=]\s*{NUMBER_PATTERN}",
        rf"\bhausdorff\s+dist\.?\s*[:=]\s*{NUMBER_PATTERN}",
        rf"\bhausdorff\s*[:=]\s*{NUMBER_PATTERN}",
        rf"\bhd95\s*[:=]\s*{NUMBER_PATTERN}",
    ],
}


# ============================================================
# Evaluation script discovery
# ============================================================

def find_evaluation_script(model_directory: Path):
    """
    Find an evaluation script inside a model directory.
    """

    for script_name in EVALUATION_SCRIPT_NAMES:
        direct_path = model_directory / script_name

        if direct_path.exists():
            return direct_path

    for script_name in EVALUATION_SCRIPT_NAMES:
        matches = list(model_directory.rglob(script_name))

        if matches:
            return matches[0]

    return None


# ============================================================
# Run an existing evaluation script
# ============================================================

def run_evaluation(model_name: str, model_directory: Path) -> str:
    """
    Run one model's existing evaluation script.

    The script is executed using the same Python interpreter used
    to launch compare_models.py.
    """

    evaluation_script = find_evaluation_script(model_directory)

    if evaluation_script is None:
        raise FileNotFoundError(
            f"No evaluation script was found in {model_directory}.\n"
            f"Expected one of: {', '.join(EVALUATION_SCRIPT_NAMES)}"
        )

    print("\n" + "=" * 75)
    print(f"Evaluating model : {model_name}")
    print(f"Model folder     : {model_directory}")
    print(f"Evaluation script: {evaluation_script}")
    print("=" * 75)

    command = [
        sys.executable,
        str(evaluation_script),
    ]

    # Make both the project root and the current model folder
    # importable by the evaluation subprocess.
    #
    # This supports root-level imports such as:
    #     from common import config
    #
    # and model-local imports such as:
    #     from model import AttentionUNet
    child_environment = os.environ.copy()

    python_paths = [
        str(PROJECT_ROOT),
        str(model_directory),
    ]

    existing_pythonpath = child_environment.get("PYTHONPATH", "")

    if existing_pythonpath:
        python_paths.append(existing_pythonpath)

    child_environment["PYTHONPATH"] = os.pathsep.join(
        python_paths
    )

    print(
        "PYTHONPATH      : "
        + child_environment["PYTHONPATH"]
    )

    process = subprocess.run(
        command,
        # Run from the project root so root-level paths and packages
        # such as common/ and datasets/ are visible.
        cwd=str(PROJECT_ROOT),
        env=child_environment,
        capture_output=True,
        text=True,
    )

    complete_output = (
        process.stdout.strip()
        + "\n"
        + process.stderr.strip()
    ).strip()

    print(complete_output)

    if process.returncode != 0:
        raise RuntimeError(
            f"{model_name} evaluation failed with exit code "
            f"{process.returncode}.\n\n{complete_output}"
        )

    return complete_output


# ============================================================
# Metric extraction
# ============================================================

def extract_metric(output_text: str, patterns):
    """
    Extract the first matching numeric value for one metric.
    """

    for pattern in patterns:
        match = re.search(
            pattern,
            output_text,
            flags=re.IGNORECASE,
        )

        if match:
            return float(match.group(1))

    return None


def extract_all_metrics(output_text: str):
    """
    Extract all supported metrics from evaluation output.
    """

    return {
        metric_name: extract_metric(output_text, patterns)
        for metric_name, patterns in METRIC_PATTERNS.items()
    }


# ============================================================
# Save evaluation logs
# ============================================================

def save_evaluation_log(model_name: str, output_text: str) -> Path:
    """
    Save complete evaluation output for debugging.
    """

    logs_directory = OUTPUT_DIRECTORY / "evaluation_logs"
    logs_directory.mkdir(parents=True, exist_ok=True)

    safe_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("-", "_")
    )

    log_path = logs_directory / f"{safe_name}.txt"

    log_path.write_text(
        output_text,
        encoding="utf-8",
    )

    return log_path


# ============================================================
# Save CSV
# ============================================================

def save_results_to_csv(results):
    """
    Save all extracted results to a CSV file.
    """

    columns = [
        "Model",
        "Loss",
        "Dice",
        "IoU",
        "Precision",
        "Recall",
        "Specificity",
        "Hausdorff Distance",
        "Status",
    ]

    with CSV_OUTPUT_PATH.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=columns,
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nCSV saved to:\n{CSV_OUTPUT_PATH}")


# ============================================================
# Plot helpers
# ============================================================

def get_usable_results(results):
    """
    Return results from evaluations that completed successfully,
    even when one or more metrics are unavailable.
    """

    return [
        result
        for result in results
        if result["Status"] in {"Success", "Partial metrics"}
    ]


def metric_has_values(results, metric_name: str) -> bool:
    """
    Check whether at least one result contains a metric value.
    """

    return any(
        result.get(metric_name) is not None
        for result in results
    )


def add_bar_labels(axis, bars, values, decimal_places=4):
    """
    Add numeric labels to bars.
    """

    labels = [
        "N/A" if value is None else f"{value:.{decimal_places}f}"
        for value in values
    ]

    axis.bar_label(
        bars,
        labels=labels,
        padding=3,
        fontsize=8,
    )


# ============================================================
# Main metrics comparison
# ============================================================

def plot_main_metrics(results):
    """
    Plot Dice, IoU, Precision, Recall and Specificity.
    """

    usable_results = get_usable_results(results)

    available_metrics = [
        metric
        for metric in MAIN_METRICS
        if metric_has_values(usable_results, metric)
    ]

    if not usable_results or not available_metrics:
        print("Main metrics graph skipped: no usable metrics found.")
        return

    model_names = [
        result["Model"]
        for result in usable_results
    ]

    x_positions = np.arange(len(model_names))

    total_group_width = 0.84
    bar_width = total_group_width / len(available_metrics)

    figure, axis = plt.subplots(
        figsize=(max(13, len(model_names) * 2.2), 8)
    )

    for metric_index, metric_name in enumerate(available_metrics):
        raw_values = [
            result.get(metric_name)
            for result in usable_results
        ]

        plotted_values = [
            0.0 if value is None else value
            for value in raw_values
        ]

        positions = (
            x_positions
            - total_group_width / 2
            + metric_index * bar_width
            + bar_width / 2
        )

        bars = axis.bar(
            positions,
            plotted_values,
            width=bar_width,
            label=metric_name,
        )

        add_bar_labels(
            axis,
            bars,
            raw_values,
            decimal_places=3,
        )

    axis.set_title(
        "Segmentation Model Performance Comparison",
        fontsize=16,
        fontweight="bold",
    )

    axis.set_xlabel("Segmentation Model")
    axis.set_ylabel("Metric Score")

    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        model_names,
        rotation=25,
        ha="right",
    )

    axis.set_ylim(0, 1.15)
    axis.legend(title="Metric")

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    figure.tight_layout()

    output_path = (
        OUTPUT_DIRECTORY
        / "all_main_metrics_comparison.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Created: {output_path}")


# ============================================================
# Single metric comparison
# ============================================================

def plot_single_metric(
    results,
    metric_name: str,
    lower_is_better: bool = False,
):
    """
    Plot one metric across all models.
    """

    usable_results = get_usable_results(results)

    metric_results = [
        result
        for result in usable_results
        if result.get(metric_name) is not None
    ]

    if not metric_results:
        print(
            f"{metric_name} graph skipped: "
            f"no values were found."
        )
        return

    metric_results = sorted(
        metric_results,
        key=lambda item: item[metric_name],
        reverse=not lower_is_better,
    )

    model_names = [
        result["Model"]
        for result in metric_results
    ]

    metric_values = [
        result[metric_name]
        for result in metric_results
    ]

    figure, axis = plt.subplots(
        figsize=(max(10, len(model_names) * 1.8), 6)
    )

    bars = axis.bar(
        model_names,
        metric_values,
    )

    add_bar_labels(
        axis,
        bars,
        metric_values,
        decimal_places=4,
    )

    direction = (
        "Lower Is Better"
        if lower_is_better
        else "Higher Is Better"
    )

    axis.set_title(
        f"{metric_name} Comparison — {direction}",
        fontsize=15,
        fontweight="bold",
    )

    axis.set_xlabel("Segmentation Model")
    axis.set_ylabel(metric_name)

    axis.tick_params(
        axis="x",
        rotation=25,
    )

    if not lower_is_better:
        axis.set_ylim(
            0,
            max(1.05, max(metric_values) + 0.1),
        )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    figure.tight_layout()

    safe_metric_name = (
        metric_name.lower()
        .replace(" ", "_")
    )

    output_path = (
        OUTPUT_DIRECTORY
        / f"{safe_metric_name}_comparison.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Created: {output_path}")


# ============================================================
# Heatmap
# ============================================================

def plot_metrics_heatmap(results):
    """
    Create a heatmap for metrics expected to lie between 0 and 1.

    Loss and Hausdorff Distance are excluded because they use
    different scales and lower values are better.
    """

    usable_results = get_usable_results(results)

    available_metrics = [
        metric
        for metric in MAIN_METRICS
        if metric_has_values(usable_results, metric)
    ]

    if not usable_results or not available_metrics:
        print("Heatmap skipped: no usable main metrics found.")
        return

    model_names = [
        result["Model"]
        for result in usable_results
    ]

    values = np.array(
        [
            [
                np.nan
                if result.get(metric) is None
                else result[metric]
                for metric in available_metrics
            ]
            for result in usable_results
        ],
        dtype=float,
    )

    figure, axis = plt.subplots(
        figsize=(
            max(9, len(available_metrics) * 1.7),
            max(5, len(model_names) * 0.8),
        )
    )

    image = axis.imshow(
        values,
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    axis.set_xticks(np.arange(len(available_metrics)))
    axis.set_xticklabels(available_metrics)

    axis.set_yticks(np.arange(len(model_names)))
    axis.set_yticklabels(model_names)

    axis.set_title(
        "Segmentation Metrics Heatmap",
        fontsize=16,
        fontweight="bold",
    )

    axis.set_xlabel("Evaluation Metric")
    axis.set_ylabel("Segmentation Model")

    for row_index in range(len(model_names)):
        for column_index in range(len(available_metrics)):
            value = values[row_index, column_index]

            label = (
                "N/A"
                if np.isnan(value)
                else f"{value:.4f}"
            )

            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
            )

    figure.colorbar(
        image,
        ax=axis,
        label="Metric Score",
    )

    figure.tight_layout()

    output_path = (
        OUTPUT_DIRECTORY
        / "segmentation_metrics_heatmap.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Created: {output_path}")


# ============================================================
# Console summary
# ============================================================

def print_results_summary(results):
    """
    Print a readable model comparison table.
    """

    print("\n" + "=" * 120)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 120)

    header = (
        f"{'Model':<18}"
        f"{'Loss':>10}"
        f"{'Dice':>10}"
        f"{'IoU':>10}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'Specificity':>14}"
        f"{'Hausdorff':>13}"
        f"{'Status':>19}"
    )

    print(header)
    print("-" * 120)

    for result in results:

        def format_value(metric_name):
            value = result.get(metric_name)

            if value is None:
                return "N/A"

            return f"{value:.4f}"

        print(
            f"{result['Model']:<18}"
            f"{format_value('Loss'):>10}"
            f"{format_value('Dice'):>10}"
            f"{format_value('IoU'):>10}"
            f"{format_value('Precision'):>12}"
            f"{format_value('Recall'):>10}"
            f"{format_value('Specificity'):>14}"
            f"{format_value('Hausdorff Distance'):>13}"
            f"{result['Status']:>19}"
        )


# ============================================================
# Evaluate all registered models
# ============================================================

def evaluate_all_models():
    """
    Run every model's existing evaluation script.
    """

    all_results = []

    for model_name, folder_name in MODEL_FOLDERS.items():
        model_directory = PROJECT_ROOT / folder_name

        base_result = {
            "Model": model_name,
            "Loss": None,
            "Dice": None,
            "IoU": None,
            "Precision": None,
            "Recall": None,
            "Specificity": None,
            "Hausdorff Distance": None,
            "Status": "Evaluation failed",
        }

        if not model_directory.exists():
            print(
                f"\nSkipping {model_name}: "
                f"folder not found at {model_directory}"
            )

            base_result["Status"] = "Folder not found"
            all_results.append(base_result)
            continue

        try:
            output_text = run_evaluation(
                model_name,
                model_directory,
            )

            log_path = save_evaluation_log(
                model_name,
                output_text,
            )

            print(f"Evaluation log saved to: {log_path}")

            extracted_metrics = extract_all_metrics(
                output_text
            )

            result = {
                "Model": model_name,
                **extracted_metrics,
                "Status": "Success",
            }

            found_count = sum(
                value is not None
                for value in extracted_metrics.values()
            )

            if found_count == 0:
                result["Status"] = "No metrics found"

            elif found_count < len(extracted_metrics):
                result["Status"] = "Partial metrics"

            all_results.append(result)

        except Exception as error:
            print(
                f"\nCould not evaluate {model_name}:\n{error}"
            )

            all_results.append(base_result)

    return all_results


# ============================================================
# Main entry point
# ============================================================

def main():
    """
    Run the complete comparison workflow.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Segmentation Model Comparison")
    print("-----------------------------")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python:       {sys.executable}")
    print(f"Output:       {OUTPUT_DIRECTORY}")

    results = evaluate_all_models()

    save_results_to_csv(results)
    print_results_summary(results)

    plot_main_metrics(results)

    for metric_name in MAIN_METRICS:
        plot_single_metric(
            results,
            metric_name,
            lower_is_better=False,
        )

    for metric_name in LOWER_IS_BETTER_METRICS:
        plot_single_metric(
            results,
            metric_name,
            lower_is_better=True,
        )

    plot_metrics_heatmap(results)

    print("\n" + "=" * 75)
    print("Comparison completed.")
    print(f"All outputs are available in:\n{OUTPUT_DIRECTORY}")
    print("=" * 75)


if __name__ == "__main__":
    main()