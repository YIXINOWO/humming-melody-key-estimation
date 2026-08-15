#!/usr/bin/env python3
"""Estimate one HumTrans time-origin correction on validation, then freeze it.

The shift is selected using validation onset-only micro F1. Test annotations are
never consulted during selection. Raw and aligned test results are both kept.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mir_eval


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revision" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jasm_revision.evaluation import evaluate_corpus  # noqa: E402
from jasm_revision.humtrans import (  # noqa: E402
    evaluate_humtrans_official,
    midi_file_to_notes,
    trim_estimates_to_reference_span,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--valid-predictions",
        type=Path,
        default=ROOT / "revision" / "results" / "humtrans_valid_zero_shot",
    )
    parser.add_argument(
        "--test-predictions",
        type=Path,
        default=ROOT / "revision" / "results" / "humtrans_zero_shot",
    )
    parser.add_argument(
        "--midi-dir",
        type=Path,
        default=ROOT / "revision" / "external" / "extracted" / "humtrans_midi",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "revision" / "results" / "humtrans_time_alignment",
    )
    parser.add_argument("--minimum-shift", type=float, default=-0.50)
    parser.add_argument("--maximum-shift", type=float, default=0.50)
    parser.add_argument("--step", type=float, default=0.01)
    return parser.parse_args()


def read_keys(split: str) -> list[str]:
    path = (
        ROOT
        / "revision"
        / "external"
        / "repos"
        / "HumTrans-main"
        / f"{split}_keys.txt"
    )
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_notes(split: str, prediction_dir: Path, midi_dir: Path) -> tuple[list[str], dict, dict]:
    keys = read_keys(split)
    references = {
        key: midi_file_to_notes(midi_dir / "GroundTruth" / split / f"{key}.mid")
        for key in keys
    }
    predictions = {}
    for key in keys:
        path = prediction_dir / f"{key}.notes.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing {split} prediction: {path}")
        predictions[key] = pd.read_csv(path)
    return keys, references, predictions


def shift_notes(notes: pd.DataFrame, seconds: float) -> pd.DataFrame:
    shifted = notes.copy()
    if shifted.empty:
        return shifted
    shifted[["onset", "offset"]] = shifted[["onset", "offset"]] + seconds
    shifted["onset"] = shifted["onset"].clip(lower=0.0)
    shifted["offset"] = shifted["offset"].clip(lower=0.0)
    return shifted.loc[
        np.isfinite(shifted["onset"])
        & np.isfinite(shifted["offset"])
        & (shifted["offset"] > shifted["onset"])
    ].reset_index(drop=True)


def aligned_predictions(predictions: dict[str, pd.DataFrame], shift: float) -> dict[str, pd.DataFrame]:
    return {key: shift_notes(notes, shift) for key, notes in predictions.items()}


def standard_metrics(
    keys: list[str], references: dict[str, pd.DataFrame], predictions: dict[str, pd.DataFrame]
) -> tuple[dict, dict, pd.DataFrame]:
    trimmed = {
        key: trim_estimates_to_reference_span(references[key], predictions[key])
        for key in keys
    }
    trimmed = {
        key: notes.loc[
            np.isfinite(notes["onset"])
            & np.isfinite(notes["offset"])
            & (notes["offset"] > notes["onset"])
        ].reset_index(drop=True)
        for key, notes in trimmed.items()
    }
    micro, per_recording, macro = evaluate_corpus(references, trimmed, keys)
    return micro, macro, per_recording


def official_macro(
    keys: list[str], references: dict[str, pd.DataFrame], predictions: dict[str, pd.DataFrame]
) -> dict[str, float]:
    rows = [evaluate_humtrans_official(references[key], predictions[key]) for key in keys]
    frame = pd.DataFrame(rows)
    return {
        "precision": float(frame["official_precision"].mean()),
        "recall": float(frame["official_recall"].mean()),
        "f1": float(frame["official_f1"].mean()),
    }


def onset_micro_f1(
    keys: list[str], references: dict[str, pd.DataFrame], predictions: dict[str, pd.DataFrame]
) -> tuple[float, int, int, int]:
    matches = 0
    reference_count = 0
    estimate_count = 0
    for key in keys:
        reference_intervals = references[key][["onset", "offset"]].to_numpy(np.float64)
        estimate_intervals = predictions[key][["onset", "offset"]].to_numpy(np.float64)
        reference_count += len(reference_intervals)
        estimate_count += len(estimate_intervals)
        if len(reference_intervals) and len(estimate_intervals):
            matches += len(
                mir_eval.transcription.match_note_onsets(
                    reference_intervals,
                    estimate_intervals,
                    onset_tolerance=0.050,
                )
            )
    precision = matches / estimate_count if estimate_count else 0.0
    recall = matches / reference_count if reference_count else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return f1, matches, reference_count, estimate_count


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    valid_keys, valid_references, valid_predictions = load_notes(
        "valid", args.valid_predictions, args.midi_dir
    )
    test_keys, test_references, test_predictions = load_notes(
        "test", args.test_predictions, args.midi_dir
    )

    shifts = np.arange(
        args.minimum_shift,
        args.maximum_shift + args.step / 2.0,
        args.step,
        dtype=np.float64,
    )
    search_rows = []
    for shift in shifts:
        aligned = aligned_predictions(valid_predictions, float(shift))
        onset_f1, matches, references, estimates = onset_micro_f1(
            valid_keys, valid_references, aligned
        )
        search_rows.append(
            {
                "shift_seconds_added_to_predictions": float(np.round(shift, 10)),
                "valid_micro_COn_F": onset_f1,
                "onset_matches": matches,
                "reference_notes": references,
                "estimated_notes": estimates,
            }
        )
    search = pd.DataFrame(search_rows)
    best_row = search.sort_values(
        ["valid_micro_COn_F", "shift_seconds_added_to_predictions"],
        ascending=[False, True],
        kind="stable",
    ).iloc[0]
    selected_shift = float(best_row["shift_seconds_added_to_predictions"])
    search.to_csv(args.output_dir / "validation_shift_search.csv", index=False)

    summary: dict[str, object] = {
        "selection_protocol": {
            "selection_split": "valid",
            "selection_metric": "standard onset-only micro COn F1",
            "candidate_min_seconds": args.minimum_shift,
            "candidate_max_seconds": args.maximum_shift,
            "candidate_step_seconds": args.step,
            "selected_seconds_added_to_predictions": selected_shift,
            "test_labels_used_for_selection": False,
            "negative_onsets_after_shift": "clipped to zero; non-positive intervals discarded",
        }
    }
    per_recording_parts = []
    for split, keys, references, predictions in (
        ("valid", valid_keys, valid_references, valid_predictions),
        ("test", test_keys, test_references, test_predictions),
    ):
        summary[split] = {}
        for condition, shift in (("raw", 0.0), ("aligned", selected_shift)):
            current = aligned_predictions(predictions, shift)
            micro, macro, per_recording = standard_metrics(keys, references, current)
            official = official_macro(keys, references, current)
            summary[split][condition] = {
                "recordings": len(keys),
                "standard_trimmed_micro": micro,
                "standard_trimmed_macro": macro,
                "official_humtrans_macro": official,
            }
            per_recording.insert(0, "condition", condition)
            per_recording.insert(0, "split", split)
            per_recording_parts.append(per_recording)

    pd.concat(per_recording_parts, ignore_index=True).to_csv(
        args.output_dir / "per_recording_metrics.csv.gz", index=False
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
