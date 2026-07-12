from __future__ import annotations

import argparse

from core import aggregate_results, get_output_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate ablation results into summary CSV files.")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Override output root. Defaults to $MTA_ABLATION_OUTPUT_ROOT or experiments/ablation/runs.",
    )
    args = parser.parse_args()

    output_root = get_output_root(args.output_root)
    outputs = aggregate_results(output_root)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
