#!/usr/bin/env python3
"""Evaluate frozen historical predictions against the corrected source GT."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revision" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jasm_revision.evaluation import evaluate_corpus  # noqa: E402
from jasm_revision.ground_truth import read_ground_truth  # noqa: E402


def main() -> int:
    output_dir = ROOT / "revision" / "results" / "historical_prediction_diagnostic"
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_paths = sorted((ROOT / "gt_files_temp").glob("*.GroundTruth.txt"))
    references = {
        path.name.removesuffix(".GroundTruth.txt"): read_ground_truth(path).notes
        for path in gt_paths
    }
    predictions = pd.read_csv(ROOT / "final_results_for_paper" / "all_predictions_onoff.csv")
    estimates = {
        stem: group.drop(columns=["stem"], errors="ignore").reset_index(drop=True)
        for stem, group in predictions.groupby("stem")
    }
    for stem in references:
        estimates.setdefault(stem, pd.DataFrame(columns=["onset", "offset", "midi", "hz"]))

    stems = sorted(references)
    micro, per_recording, macro = evaluate_corpus(references, estimates, stems)
    per_recording.to_csv(output_dir / "per_recording_corrected_gt.csv", index=False)
    summary = {
        "purpose": "diagnostic only; frozen historical predictions, corrected GT, no retraining",
        "micro": micro,
        "macro": macro,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

