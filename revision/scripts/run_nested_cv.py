#!/usr/bin/env python3
"""CLI for the audited Molina grouped nested-CV experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revision" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jasm_revision.nested_cv import run_nested_cv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="use two outer/inner folds, 20-tree RF, and two decoder candidates",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = run_nested_cv(
        ROOT,
        ROOT / "revision" / "config" / "main_experiment.yaml",
        smoke_test=args.smoke_test,
        overwrite=args.overwrite,
    )
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

