"""
Process Quarto documents and notebooks in this repository.

Default behavior:
  - Render all `.qmd` files under `examples/` with `quarto render`
  - Execute all `.ipynb` files under `application/` and `examples/` in place
    with `python -m jupyter nbconvert --to notebook --execute --inplace`

Examples:
  python scripts/process_documents.py
  python scripts/process_documents.py --qmd-only
  python scripts/process_documents.py --notebooks-only
  python scripts/process_documents.py --files examples/mixed_type_example.qmd application/ema_affect.ipynb
  python scripts/process_documents.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIRS = ("examples", "application")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render repository QMD files and execute notebooks."
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional explicit files to process. Defaults to discovery in examples/ and application/.",
    )
    parser.add_argument(
        "--qmd-only",
        action="store_true",
        help="Process only .qmd files.",
    )
    parser.add_argument(
        "--notebooks-only",
        action="store_true",
        help="Process only .ipynb files.",
    )
    parser.add_argument(
        "--notebook-mode",
        choices=("execute", "render"),
        default="execute",
        help="How to process notebooks: execute in place with nbconvert, or render with Quarto.",
    )
    parser.add_argument(
        "--notebook-timeout",
        type=int,
        default=600,
        help="Execution timeout in seconds when --notebook-mode execute is used.",
    )
    parser.add_argument(
        "--quarto-bin",
        default="quarto",
        help="Quarto executable to use for .qmd rendering and notebook rendering.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable to use for notebook execution.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep processing remaining files after a failure.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without executing them.",
    )
    return parser


def discover_files(explicit_files: list[str] | None) -> list[Path]:
    if explicit_files:
        paths = [(ROOT / Path(raw)).resolve() if not Path(raw).is_absolute() else Path(raw) for raw in explicit_files]
        return sorted(paths)

    files: list[Path] = []
    for rel_dir in SEARCH_DIRS:
        directory = ROOT / rel_dir
        if not directory.exists():
            continue
        files.extend(sorted(directory.rglob("*.qmd")))
        files.extend(sorted(directory.rglob("*.ipynb")))
    return sorted(files)


def should_process(path: Path, args: argparse.Namespace) -> bool:
    if args.qmd_only:
        return path.suffix.lower() == ".qmd"
    if args.notebooks_only:
        return path.suffix.lower() == ".ipynb"
    return path.suffix.lower() in {".qmd", ".ipynb"}


def build_command(path: Path, args: argparse.Namespace) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".qmd":
        return [args.quarto_bin, "render", str(path)]
    if suffix == ".ipynb":
        if args.notebook_mode == "render":
            return [args.quarto_bin, "render", str(path)]
        return [
            args.python_bin,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            f"--ExecutePreprocessor.timeout={args.notebook_timeout}",
            str(path),
        ]
    raise ValueError(f"Unsupported file type: {path}")


def run_command(command: list[str], dry_run: bool) -> int:
    printable = subprocess.list2cmdline(command)
    print(f"$ {printable}")
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return int(completed.returncode)


def main() -> int:
    args = build_parser().parse_args()
    if args.qmd_only and args.notebooks_only:
        raise SystemExit("Choose at most one of --qmd-only and --notebooks-only.")

    files = [path for path in discover_files(args.files) if should_process(path, args)]
    if not files:
        print("No matching .qmd or .ipynb files found.")
        return 0

    failures: list[tuple[Path, int]] = []
    print(f"Processing {len(files)} file(s) from {ROOT}")
    for path in files:
        command = build_command(path, args)
        code = run_command(command, dry_run=args.dry_run)
        if code != 0:
            failures.append((path, code))
            print(f"FAILED: {path} (exit code {code})")
            if not args.continue_on_error:
                break

    succeeded = len(files) - len(failures)
    print(f"Finished. Succeeded: {succeeded}, Failed: {len(failures)}")
    if failures:
        for path, code in failures:
            print(f"  - {path}: exit code {code}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
