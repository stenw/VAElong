"""
CLI entry point for YAML-driven VAElong applications.
"""

from __future__ import annotations

import argparse

from .app_config import load_app_config
from .app_runner import run_application


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a YAML-configured VAElong application.")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--data-path", default=None, help="Optional override for the configured input data path.")
    parser.add_argument("--output-dir", default=None, help="Optional override for the configured output directory.")
    parser.add_argument(
        "--plot-ids",
        nargs="*",
        default=None,
        help="Optional override for subject ids to plot. If omitted, the config value is used.",
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
