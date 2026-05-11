"""
Thin entry point for the YAML-configured glucose landmark application.

The actual modelling pipeline lives in `vaelong.app_runner` and is configured
through `configs/glucose.yaml`. This wrapper keeps the old application path
available while avoiding hard-coded analysis logic in the script itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vaelong.app_config import load_app_config
from vaelong.app_runner import run_application


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "glucose.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the glucose landmark application from a YAML config."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML configuration file. Defaults to configs/glucose.yaml.",
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
