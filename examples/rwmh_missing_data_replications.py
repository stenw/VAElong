"""
Repeated simulation study for the missing-data benchmark.

Runs the focused RWMH-vs-RNN-vs-mixed-model simulation repeatedly with new
synthetic datasets and writes per-replication and summary CSV outputs that can
be explored in a notebook or Quarto document.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rwmh_missing_data_benchmark import run_benchmark


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "rwmh_missing_data_replications_files"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the missing-data benchmark repeatedly on simulated datasets."
    )
    parser.add_argument("--n-replications", type=int, default=100, help="Number of simulated datasets to generate.")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed; replication i uses base_seed + i.")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of subjects per replication.")
    parser.add_argument("--seq-len", type=int, default=50, help="Sequence length per subject.")
    parser.add_argument("--n-baseline", type=int, default=3, help="Number of baseline covariates.")
    parser.add_argument("--missing-rate", type=float, default=0.15, help="Random missingness rate.")
    parser.add_argument(
        "--missing-pattern",
        choices=("random", "block", "monotone"),
        default="random",
        help="Missingness pattern used when masking the simulated data.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to store CSV summaries and metadata.",
    )
    return parser


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [col for col in ["RMSE", "Corr", "LogLik", "AUC"] if col in results_df.columns]
    grouped = (
        results_df
        .groupby(["Model", "Variable"], dropna=False)[metrics]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(str(part) for part in col if part).rstrip("_")
        if isinstance(col, tuple) else str(col)
        for col in grouped.columns
    ]
    return grouped


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    replication_rows = []
    hyperparameter_rows = []
    for replication in range(args.n_replications):
        seed = args.base_seed + replication
        print(
            f"\n=== Replication {replication + 1}/{args.n_replications} "
            f"(seed={seed}) ==="
        )
        result = run_benchmark(
            show_plots=False,
            seed=seed,
            n_samples=args.n_samples,
            seq_len=args.seq_len,
            n_baseline=args.n_baseline,
            missing_rate=args.missing_rate,
            missing_pattern=args.missing_pattern,
            output_dir=output_dir / f"rep_{replication:03d}",
            save_artifacts=False,
            verbose=False,
        )

        result_df = result["results_df"].copy()
        result_df.insert(0, "Replication", replication)
        result_df.insert(1, "Seed", seed)
        replication_rows.append(result_df)

        vae_hp = result.get("best_hp", {})
        rnn_hp = result.get("rnn_best_hp", {})
        hyperparameter_rows.append(
            {
                "Replication": replication,
                "Seed": seed,
                "VAE_learning_rate": vae_hp.get("learning_rate"),
                "VAE_weight_decay": vae_hp.get("weight_decay"),
                "RNN_hidden_dim": rnn_hp.get("hidden_dim"),
                "RNN_learning_rate": rnn_hp.get("learning_rate"),
            }
        )

    replications_df = pd.concat(replication_rows, ignore_index=True)
    summary_df = summarize_results(replications_df)
    hyperparameters_df = pd.DataFrame(hyperparameter_rows)

    replications_df.to_csv(output_dir / "replication_results.csv", index=False)
    summary_df.to_csv(output_dir / "replication_summary.csv", index=False)
    hyperparameters_df.to_csv(output_dir / "replication_hyperparameters.csv", index=False)

    metadata = {
        "n_replications": args.n_replications,
        "base_seed": args.base_seed,
        "n_samples": args.n_samples,
        "seq_len": args.seq_len,
        "n_baseline": args.n_baseline,
        "missing_rate": args.missing_rate,
        "missing_pattern": args.missing_pattern,
    }
    (output_dir / "replication_config.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:")
    print(f"  {output_dir / 'replication_results.csv'}")
    print(f"  {output_dir / 'replication_summary.csv'}")
    print(f"  {output_dir / 'replication_hyperparameters.csv'}")
    print(f"  {output_dir / 'replication_config.json'}")

    print("\nSummary (means by model and variable):")
    mean_cols = [col for col in summary_df.columns if col.endswith("_mean")]
    preview_cols = ["Model", "Variable"] + mean_cols
    print(summary_df[preview_cols].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))


if __name__ == "__main__":
    main()
