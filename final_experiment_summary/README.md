# Final Relevant Experiment Summary Utility

Place the `final_experiment_summary` folder directly inside the
`Ovarian_Cancer` project root.

## Option 1: Collect results already generated

If every experiment has already been evaluated and its `metric_summary.csv`
exists, run:

```bash
python final_experiment_summary/final_experiment_summary.py
```

This is the recommended first command because it is fast and does not rerun
model inference.

## Option 2: Run every relevant evaluator and summarize

```bash
caffeinate -i python final_experiment_summary/final_experiment_summary.py \
  --evaluate
```

An evaluation failure or missing checkpoint is recorded in
`skipped_experiments.csv`; it does not stop the remaining experiments.

## Summarize only the controlled ablations

```bash
python final_experiment_summary/final_experiment_summary.py \
  --groups ablation
```

## Summarize selected groups and rerun their evaluators

```bash
caffeinate -i python final_experiment_summary/final_experiment_summary.py \
  --evaluate \
  --groups binary context ablation
```

## Outputs

Files are written to:

```text
classification_results/final_relevant_experiment_summary/
```

The output includes:

```text
all_relevant_experiments.csv
multiclass_experiments.csv
binary_experiments.csv
context_experiments.csv
ablation_experiments.csv
skipped_experiments.csv
final_experiment_summary.md
multiseed_mean_std.csv (when multi-seed runs are complete)
```

The Markdown file contains grouped, paper-ready comparison tables. The CSV
files retain all normalized metrics found in each experiment's summary.
When the multi-seed summary exists, it is automatically incorporated into the
final Markdown report and copied into the final summary directory.

## Included experiment families

- Multiclass mask-aware model evolution
- Main binary models
- Full-image context experiments
- Controlled ablations A1 and A3-A7 plus the final reference

The script intentionally excludes segmentation architectures and unrelated
standalone classifiers because their metrics and research questions belong in
separate comparison tables.
