"""
Thin entry point for the YAML-configured EMA VAE application.

This runs the VAE-only EMA workflow from `configs/ema_vae.yaml`. The legacy
`application/ema_affect.py` script remains the place for the custom mixed-model
benchmark until that benchmark is generalized.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vaelong.app_config import load_app_config
from vaelong.app_runner import run_application


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "ema_vae.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the EMA VAE application from a YAML config."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML configuration file. Defaults to configs/ema_vae.yaml.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional override for the configured input data path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for the configured output directory.",
    )
    parser.add_argument(
        "--plot-ids",
        nargs="*",
        default=None,
        help="Optional subject ids to plot. If omitted, random test subjects are used.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_app_config(args.config)
    run_application(
        config=config,
        data_path_override=args.data_path,
        output_dir_override=args.output_dir,
        plot_ids_override=args.plot_ids,
    )


if __name__ == "__main__":
    main()
