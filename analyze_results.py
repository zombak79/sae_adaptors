"""Collect completed experiment metrics into a flat results table."""

import argparse
from pathlib import Path

from utils import build_checkpoints, gather_results, load_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.py"),
        help="Path to a Python experiment configuration file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV file to write in addition to printing the results.",
    )
    args = parser.parse_args()

    config = load_experiment_config(["--config", str(args.config)])
    checkpoints = build_checkpoints(config.CONFIG, config.CHECKPOINT_PATH_PREFIX)
    results = gather_results(checkpoints)

    if not results.empty:
        results["sbert_name"] = results["sbert_name"].str.replace("/", "_", regex=False)
        results = results.sort_values(
            ["dataset", "sbert_name", "base_stage", "model"],
            na_position="first",
        )

    print(results.to_string(index=False))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
