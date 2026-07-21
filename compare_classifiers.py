from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_FILES = {
    "ResNet50": "resnet50_evaluation_metrics.json",
    "EfficientNet-B3": "efficientnet_b3_evaluation_metrics.json",
    "DenseNet121": "densenet121_evaluation_metrics.json",
    "ViT-B/16": "vit_b16_evaluation_metrics.json",
    "Swin-T": "swin_t_evaluation_metrics.json",
}


def load_classifier_results(
    metrics_dir: Path,
) -> pd.DataFrame:
    rows = []

    for model_name, file_name in MODEL_FILES.items():
        metrics_path = metrics_dir / file_name

        if not metrics_path.exists():
            print(
                f"Skipping {model_name}: "
                f"{metrics_path} was not found"
            )
            continue

        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)

        rows.append(
            {
                "Model": model_name,
                "Accuracy": metrics.get("accuracy"),
                "Balanced Accuracy": metrics.get(
                    "balanced_accuracy"
                ),
                "Macro Precision": metrics.get(
                    "macro_precision"
                ),
                "Macro Recall": metrics.get("macro_recall"),
                "Macro F1": metrics.get("macro_f1"),
                "Weighted F1": metrics.get("weighted_f1"),
                "Top-3 Accuracy": metrics.get(
                    "top_3_accuracy"
                ),
                "Macro ROC-AUC": metrics.get(
                    "macro_roc_auc_ovr"
                ),
            }
        )

    if not rows:
        raise FileNotFoundError(
            "No classifier evaluation metric files were found in "
            f"{metrics_dir}"
        )

    dataframe = pd.DataFrame(rows)

    numeric_columns = [
        column
        for column in dataframe.columns
        if column != "Model"
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    return dataframe


def save_metric_comparison_chart(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save a grouped bar chart for the most important metrics.
    """
    selected_metrics = [
        "Accuracy",
        "Balanced Accuracy",
        "Macro F1",
        "Weighted F1",
    ]

    available_metrics = [
        metric
        for metric in selected_metrics
        if metric in dataframe.columns
        and dataframe[metric].notna().any()
    ]

    if not available_metrics:
        print(
            "Skipping metric comparison chart: "
            "no valid metrics were found."
        )
        return

    chart_data = dataframe[
        ["Model", *available_metrics]
    ].copy()

    chart_data = chart_data.set_index("Model")

    axis = chart_data.plot(
        kind="bar",
        figsize=(14, 8),
        width=0.80,
    )

    axis.set_title(
        "Ovarian Tumor Classifier Metric Comparison"
    )
    axis.set_xlabel("Classifier")
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.05)
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )
    axis.legend(
        title="Metric",
        loc="lower right",
    )

    plt.xticks(
        rotation=25,
        ha="right",
    )
    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_macro_f1_ranking_chart(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save a horizontal bar chart ranking models by Macro F1.
    """
    chart_data = dataframe[
        ["Model", "Macro F1"]
    ].dropna()

    if chart_data.empty:
        print(
            "Skipping Macro F1 ranking chart: "
            "no Macro F1 values were found."
        )
        return

    chart_data = chart_data.sort_values(
        "Macro F1",
        ascending=True,
    )

    plt.figure(figsize=(11, 7))

    bars = plt.barh(
        chart_data["Model"],
        chart_data["Macro F1"],
    )

    plt.title("Classifier Ranking by Macro F1")
    plt.xlabel("Macro F1 Score")
    plt.ylabel("Classifier")
    plt.xlim(0.0, 1.05)
    plt.grid(
        axis="x",
        linestyle="--",
        alpha=0.4,
    )

    for bar, score in zip(
        bars,
        chart_data["Macro F1"],
    ):
        plt.text(
            score + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}",
            va="center",
        )

    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_accuracy_macro_f1_scatter(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save a scatter plot comparing accuracy and Macro F1.
    """
    chart_data = dataframe[
        ["Model", "Accuracy", "Macro F1"]
    ].dropna()

    if chart_data.empty:
        print(
            "Skipping scatter plot: "
            "accuracy or Macro F1 values are missing."
        )
        return

    plt.figure(figsize=(10, 7))

    plt.scatter(
        chart_data["Accuracy"],
        chart_data["Macro F1"],
        s=130,
    )

    for _, row in chart_data.iterrows():
        plt.annotate(
            row["Model"],
            (
                row["Accuracy"],
                row["Macro F1"],
            ),
            xytext=(7, 7),
            textcoords="offset points",
        )

    plt.title("Accuracy vs Macro F1")
    plt.xlabel("Accuracy")
    plt.ylabel("Macro F1")
    plt.xlim(0.0, 1.05)
    plt.ylim(0.0, 1.05)
    plt.grid(
        linestyle="--",
        alpha=0.4,
    )
    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    results_dir = Path("classification_results")
    metrics_dir = results_dir / "metrics"
    charts_dir = results_dir / "charts"

    dataframe = load_classifier_results(
        metrics_dir=metrics_dir,
    )

    dataframe = dataframe.sort_values(
        by="Macro F1",
        ascending=False,
        na_position="last",
    )

    csv_path = (
        results_dir
        / "classifier_comparison.csv"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        csv_path,
        index=False,
    )

    save_metric_comparison_chart(
        dataframe=dataframe,
        output_path=(
            charts_dir
            / "classifier_metric_comparison.png"
        ),
    )

    save_macro_f1_ranking_chart(
        dataframe=dataframe,
        output_path=(
            charts_dir
            / "classifier_macro_f1_ranking.png"
        ),
    )

    save_accuracy_macro_f1_scatter(
        dataframe=dataframe,
        output_path=(
            charts_dir
            / "classifier_accuracy_vs_macro_f1.png"
        ),
    )

    print("\nClassifier Comparison")
    print("=" * 120)

    print(
        dataframe.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nSaved outputs")
    print("-" * 120)
    print(f"CSV             : {csv_path}")
    print(
        "Metric chart    : "
        f"{charts_dir / 'classifier_metric_comparison.png'}"
    )
    print(
        "Macro F1 chart  : "
        f"{charts_dir / 'classifier_macro_f1_ranking.png'}"
    )
    print(
        "Scatter chart   : "
        f"{charts_dir / 'classifier_accuracy_vs_macro_f1.png'}"
    )


if __name__ == "__main__":
    main()