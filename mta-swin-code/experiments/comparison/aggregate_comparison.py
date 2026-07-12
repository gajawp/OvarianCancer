from __future__ import annotations

import argparse
import math
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG


def _parse_misclassified(mis_str: str):
    try:
        misclassified, total = str(mis_str).strip().split("/")
        return int(float(misclassified)), int(float(total))
    except Exception:
        return float("nan"), float("nan")


def bootstrap_ci_mean_accuracy_across_seeds(
    y_true: np.ndarray,
    y_pred_list: list[np.ndarray],
    n_boot: int,
    ci: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    num_samples = len(y_true)
    if num_samples == 0 or not y_pred_list:
        return float("nan"), float("nan")

    accuracy_samples = np.empty(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        sampled_indices = rng.integers(0, num_samples, size=num_samples)
        sampled_truth = y_true[sampled_indices]
        per_seed_accuracies = [
            np.mean(np.asarray(predictions)[sampled_indices] == sampled_truth)
            for predictions in y_pred_list
        ]
        accuracy_samples[idx] = float(np.mean(per_seed_accuracies))

    alpha = (100.0 - ci) / 2.0
    return (
        float(np.percentile(accuracy_samples, alpha)),
        float(np.percentile(accuracy_samples, 100.0 - alpha)),
    )


def build_final_summary_table(
    all_seed_dfs: list[pd.DataFrame],
    pred_store: list[dict],
    n_boot: int,
    ci: float,
    bootstrap_seed: int,
):
    seed_level_df = pd.concat(all_seed_dfs, ignore_index=True)
    grouped = seed_level_df.groupby("Model", as_index=False)

    acc_mean = grouped["Accuracy (%)"].mean().rename(columns={"Accuracy (%)": "Acc_mean(%)"})
    acc_std = grouped["Accuracy (%)"].std(ddof=1).rename(columns={"Accuracy (%)": "Acc_std(%)"})
    bal_mean = grouped["Balanced Accuracy (%)"].mean().rename(columns={"Balanced Accuracy (%)": "Balanced Acc(%)"})
    mcc_mean = grouped["MCC"].mean().rename(columns={"MCC": "MCC"})
    auc_mean = grouped["ROC-AUC (%)"].mean().rename(columns={"ROC-AUC (%)": "ROC-AUC(%)"})
    macro_f1_mean = grouped["Macro F1 (%)"].mean().rename(columns={"Macro F1 (%)": "Macro F1(%)"})
    prec_mean = grouped["Precision (%)"].mean().rename(columns={"Precision (%)": "Precision(%)"})
    sens_mean = grouped["Sensitivity (%)"].mean().rename(columns={"Sensitivity (%)": "Sensitivity(%)"})
    spec_mean = grouped["Specificity (%)"].mean().rename(columns={"Specificity (%)": "Specificity(%)"})
    time_mean = grouped["Time/Epoch (s)"].mean().rename(columns={"Time/Epoch (s)": "Time/Epoch (s)"})
    flops_first = grouped["FLOPs (G)"].first().rename(columns={"FLOPs (G)": "FLOPs(G)"})
    mem_first = grouped["Peak GPU Mem (MB)"].first().rename(columns={"Peak GPU Mem (MB)": "PeakMem(MB)"})
    thr_first = grouped["Throughput (imgs/s@bs=32)"].first().rename(
        columns={"Throughput (imgs/s@bs=32)": "Imgs/s@bs32"}
    )

    params_first = grouped["Parameters"].first().rename(columns={"Parameters": "Parameters"})
    params_first["Parameters(M)"] = params_first["Parameters"].map(
        lambda value: float(value) / 1e6 if pd.notnull(value) else float("nan")
    )
    params_first = params_first[["Model", "Parameters(M)"]]

    mis_rows = []
    for model_name, sub_df in seed_level_df.groupby("Model"):
        mis_values = []
        total_ref = None
        for value in sub_df["Misclassified (n/total)"].tolist():
            misclassified, total = _parse_misclassified(value)
            if not (isinstance(misclassified, float) and math.isnan(misclassified)):
                mis_values.append(misclassified)
                if total_ref is None and not (isinstance(total, float) and math.isnan(total)):
                    total_ref = total
        if not mis_values or total_ref is None:
            mis_rows.append((model_name, "nan/nan"))
        else:
            mis_rows.append((model_name, f"{float(np.mean(mis_values)):.1f}/{int(total_ref)}"))
    mis_df = pd.DataFrame(mis_rows, columns=["Model", "Misclassified (n/total)"])

    summary_df = (
        acc_mean.merge(acc_std, on="Model", how="left")
        .merge(bal_mean, on="Model", how="left")
        .merge(mcc_mean, on="Model", how="left")
        .merge(auc_mean, on="Model", how="left")
        .merge(macro_f1_mean, on="Model", how="left")
        .merge(prec_mean, on="Model", how="left")
        .merge(sens_mean, on="Model", how="left")
        .merge(spec_mean, on="Model", how="left")
        .merge(flops_first, on="Model", how="left")
        .merge(mem_first, on="Model", how="left")
        .merge(thr_first, on="Model", how="left")
        .merge(params_first, on="Model", how="left")
        .merge(time_mean, on="Model", how="left")
        .merge(mis_df, on="Model", how="left")
    )

    by_model: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)
    for record in pred_store:
        by_model[record["Model"]].append(
            (
                int(record["Seed"]),
                np.asarray(record["y_true"]),
                np.asarray(record["y_pred"]),
            )
        )

    ci_lows = []
    ci_highs = []
    for model_name in summary_df["Model"].tolist():
        runs = sorted(by_model.get(model_name, []), key=lambda item: item[0])
        if not runs:
            ci_lows.append(float("nan"))
            ci_highs.append(float("nan"))
            continue

        y_true = runs[0][1]
        y_preds = [run[2] for run in runs]
        ci_low, ci_high = bootstrap_ci_mean_accuracy_across_seeds(
            y_true=y_true,
            y_pred_list=y_preds,
            n_boot=n_boot,
            ci=ci,
            seed=bootstrap_seed,
        )
        ci_lows.append(ci_low * 100.0)
        ci_highs.append(ci_high * 100.0)

    summary_df["Acc_CI_low(%)"] = ci_lows
    summary_df["Acc_CI_high(%)"] = ci_highs
    summary_df["Accuracy (mean ± std, 3 seeds)"] = summary_df.apply(
        lambda row: f"{row['Acc_mean(%)']:.2f} ± {row['Acc_std(%)']:.2f}",
        axis=1,
    )
    summary_df["Accuracy 95% CI (bootstrap on test)"] = summary_df.apply(
        lambda row: f"[{row['Acc_CI_low(%)']:.2f}, {row['Acc_CI_high(%)']:.2f}]",
        axis=1,
    )

    final_columns = [
        "Model",
        "Accuracy (mean ± std, 3 seeds)",
        "Accuracy 95% CI (bootstrap on test)",
        "Balanced Acc(%)",
        "MCC",
        "ROC-AUC(%)",
        "Macro F1(%)",
        "Precision(%)",
        "Sensitivity(%)",
        "Specificity(%)",
        "FLOPs(G)",
        "PeakMem(MB)",
        "Imgs/s@bs32",
        "Parameters(M)",
        "Time/Epoch (s)",
        "Misclassified (n/total)",
    ]
    final_summary = summary_df[final_columns].copy()

    round_columns = [
        "Balanced Acc(%)",
        "MCC",
        "ROC-AUC(%)",
        "Macro F1(%)",
        "Precision(%)",
        "Sensitivity(%)",
        "Specificity(%)",
        "FLOPs(G)",
        "PeakMem(MB)",
        "Imgs/s@bs32",
        "Parameters(M)",
        "Time/Epoch (s)",
    ]
    for column in round_columns:
        final_summary[column] = final_summary[column].map(
            lambda value: round(float(value), 3) if pd.notnull(value) else value
        )

    return final_summary, seed_level_df


def find_latest_seed_artifacts(seed_runs_dir: Path, seeds: list[int]):
    metrics_paths = []
    preds_paths = []
    for seed in seeds:
        metric_candidates = sorted(seed_runs_dir.glob(f"seed{seed}_metrics_*.csv"))
        pred_candidates = sorted(seed_runs_dir.glob(f"seed{seed}_preds_*.pkl"))
        if not metric_candidates:
            raise FileNotFoundError(f"No metric CSV found for seed {seed} in {seed_runs_dir}")
        if not pred_candidates:
            raise FileNotFoundError(f"No prediction PKL found for seed {seed} in {seed_runs_dir}")
        metrics_paths.append(metric_candidates[-1])
        preds_paths.append(pred_candidates[-1])
    return metrics_paths, preds_paths


def load_seed_artifacts(metrics_paths: list[Path], preds_paths: list[Path]):
    all_seed_dfs = [pd.read_csv(path) for path in metrics_paths]
    pred_store = []
    for path in preds_paths:
        with path.open("rb") as handle:
            pred_store.extend(pickle.load(handle))
    return all_seed_dfs, pred_store


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate seed-level comparison artifacts.")
    parser.add_argument(
        "--seed-runs-dir",
        type=Path,
        default=DEFAULT_CONFIG.seed_run_dir,
        help="Directory containing seed*_metrics_*.csv and seed*_preds_*.pkl outputs.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=list(DEFAULT_CONFIG.default_seeds),
        help="Seeds to aggregate when auto-discovering artifacts.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        nargs="*",
        help="Optional explicit metric CSV paths. If provided, --pred-pkls must also be provided.",
    )
    parser.add_argument(
        "--pred-pkls",
        type=Path,
        nargs="*",
        help="Optional explicit prediction PKL paths. If provided, --metrics-csv must also be provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CONFIG.final_table_dir,
        help="Directory for final summary outputs.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=DEFAULT_CONFIG.bootstrap_iterations,
        help="Bootstrap iterations for confidence intervals.",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=DEFAULT_CONFIG.confidence_interval,
        help="Confidence interval percentage.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_CONFIG.bootstrap_seed,
        help="Random seed for bootstrap resampling.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if bool(args.metrics_csv) != bool(args.pred_pkls):
        raise ValueError("--metrics-csv and --pred-pkls must be provided together.")

    if args.metrics_csv and args.pred_pkls:
        if len(args.metrics_csv) != len(args.pred_pkls):
            raise ValueError("The number of metric CSVs and prediction PKLs must match.")
        metrics_paths = args.metrics_csv
        preds_paths = args.pred_pkls
    else:
        metrics_paths, preds_paths = find_latest_seed_artifacts(args.seed_runs_dir, args.seeds)

    print("Using metric CSVs:")
    for path in metrics_paths:
        print(f"  - {path}")
    print("Using prediction PKLs:")
    for path in preds_paths:
        print(f"  - {path}")

    all_seed_dfs, pred_store = load_seed_artifacts(metrics_paths, preds_paths)
    print("Loaded seed-level metric CSVs:", [len(df) for df in all_seed_dfs])
    print("Loaded pred_store runs:", len(pred_store))
    print("Models in pred_store:", sorted({record["Model"] for record in pred_store}))
    print("Seeds in pred_store:", sorted({int(record["Seed"]) for record in pred_store}))

    final_summary_df, seed_level_df = build_final_summary_table(
        all_seed_dfs=all_seed_dfs,
        pred_store=pred_store,
        n_boot=args.n_boot,
        ci=args.ci,
        bootstrap_seed=args.bootstrap_seed,
    )

    print("\nFinal summary:")
    print(final_summary_df.to_string(index=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_summary_path = args.output_dir / f"final_summary_{timestamp}.csv"
    seed_level_path = args.output_dir / f"seed_level_{timestamp}.csv"
    final_summary_df.to_csv(final_summary_path, index=False)
    seed_level_df.to_csv(seed_level_path, index=False)
    print(f"\nSaved final summary CSV: {final_summary_path}")
    print(f"Saved seed-level CSV: {seed_level_path}")


if __name__ == "__main__":
    main()
