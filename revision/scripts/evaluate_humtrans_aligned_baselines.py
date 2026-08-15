#!/usr/bin/env python3
"""Validation-aligned comparison of official HumTrans baseline predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mir_eval
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revision" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jasm_revision.evaluation import evaluate_corpus  # noqa: E402
from jasm_revision.humtrans import midi_file_to_notes, trim_estimates_to_reference_span  # noqa: E402


MODELS = ["VOCANO", "SheetSage", "MIR-ST500", "JDC-STP"]
MIDI_DIR = ROOT / "revision" / "external" / "extracted" / "humtrans_midi"
REPO = ROOT / "revision" / "external" / "repos" / "HumTrans-main"
OUTPUT = ROOT / "revision" / "results" / "humtrans_aligned_baselines"


def keys(split: str) -> list[str]:
    return [line.strip() for line in (REPO / f"{split}_keys.txt").read_text(encoding="utf-8").splitlines() if line.strip()]


def load(split: str, model: str) -> tuple[list[str], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    ordered = keys(split)
    reference = {key: midi_file_to_notes(MIDI_DIR / "GroundTruth" / split / f"{key}.mid") for key in ordered}
    estimate = {key: midi_file_to_notes(MIDI_DIR / model / split / f"{key}.mid") for key in ordered}
    return ordered, reference, estimate


def shift_notes(notes: pd.DataFrame, seconds: float) -> pd.DataFrame:
    shifted = notes.copy()
    if shifted.empty:
        return shifted
    shifted[["onset", "offset"]] += seconds
    shifted["onset"] = shifted["onset"].clip(lower=0.0)
    shifted["offset"] = shifted["offset"].clip(lower=0.0)
    return shifted.loc[shifted["offset"] > shifted["onset"]].reset_index(drop=True)


def onset_counts(ordered, reference, estimate, shift):
    matches = refs = ests = 0
    for key in ordered:
        ref_intervals = reference[key][["onset", "offset"]].to_numpy(np.float64)
        shifted = shift_notes(estimate[key], shift)
        est_intervals = shifted[["onset", "offset"]].to_numpy(np.float64)
        refs += len(ref_intervals)
        ests += len(est_intervals)
        if len(ref_intervals) and len(est_intervals):
            matches += len(mir_eval.transcription.match_note_onsets(ref_intervals, est_intervals, onset_tolerance=0.050))
    precision = matches / ests if ests else 0.0
    recall = matches / refs if refs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f1, matches, refs, ests


def evaluate(ordered, reference, estimate, shift):
    aligned = {
        key: trim_estimates_to_reference_span(reference[key], shift_notes(estimate[key], shift))
        for key in ordered
    }
    micro, per_recording, macro = evaluate_corpus(reference, aligned, ordered)
    return micro, macro, per_recording


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shifts = np.arange(-0.50, 0.5001, 0.01)
    summaries = []
    search_parts = []
    per_parts = []
    for model in MODELS:
        valid_keys, valid_ref, valid_est = load("valid", model)
        rows = []
        for shift in shifts:
            f1, matches, refs, ests = onset_counts(valid_keys, valid_ref, valid_est, float(shift))
            rows.append({"model": model, "shift_seconds_added_to_predictions": float(np.round(shift, 10)), "valid_micro_COn_F": f1, "matches": matches, "ref_notes": refs, "est_notes": ests})
        search = pd.DataFrame(rows)
        selected = float(search.sort_values(["valid_micro_COn_F", "shift_seconds_added_to_predictions"], ascending=[False, True], kind="stable").iloc[0]["shift_seconds_added_to_predictions"])
        search_parts.append(search)
        for split in ("valid", "test"):
            ordered, reference, estimate = load(split, model)
            for condition, shift in (("raw", 0.0), ("aligned", selected)):
                micro, macro, per = evaluate(ordered, reference, estimate, shift)
                summaries.append({"model": model, "split": split, "condition": condition, "selected_validation_shift_seconds": selected, "recordings": len(ordered), "standard_micro": micro, "standard_macro": macro})
                per.insert(0, "condition", condition)
                per.insert(0, "split", split)
                per.insert(0, "model", model)
                per_parts.append(per)
        print(model, selected, summaries[-1]["standard_micro"], flush=True)
    pd.concat(search_parts, ignore_index=True).to_csv(OUTPUT / "validation_shift_search.csv.gz", index=False)
    pd.concat(per_parts, ignore_index=True).to_csv(OUTPUT / "per_recording_metrics.csv.gz", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps({"selection_metric": "validation standard onset-only micro COn F1", "test_labels_used_for_selection": False, "evaluations": summaries}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
