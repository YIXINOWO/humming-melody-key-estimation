"""Note-level evaluation utilities shared by all revision experiments."""

from __future__ import annotations

from collections.abc import Mapping

import mir_eval
import numpy as np
import pandas as pd


ONSET_TOLERANCE = 0.050
PITCH_TOLERANCE_CENTS = 50.0
OFFSET_RATIO = 0.20
OFFSET_MIN_TOLERANCE = 0.050


def midi_to_hz(values: np.ndarray | pd.Series | float) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    return 440.0 * np.power(2.0, (values_array - 69.0) / 12.0)


def _intervals_and_hz(notes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if notes.empty:
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.float64)
    intervals = notes[["onset", "offset"]].to_numpy(dtype=np.float64)
    hz = (
        notes["hz"].to_numpy(dtype=np.float64)
        if "hz" in notes
        else midi_to_hz(notes["midi"])
    )
    return intervals, hz


def segmentation_error_rates(
    reference_intervals: np.ndarray, estimated_intervals: np.ndarray
) -> tuple[float, float, float]:
    """Return the historical overlap-count approximation for audit continuity.

    The revised paper will distinguish this implementation from any official
    Molina evaluation code until equivalence has been verified.
    """

    denominator = max(len(reference_intervals), 1)
    if len(estimated_intervals) == 0:
        return 0.0, 0.0, 0.0
    overlap = np.zeros((len(reference_intervals), len(estimated_intervals)), dtype=bool)
    for ref_index, (ref_onset, ref_offset) in enumerate(reference_intervals):
        overlap[ref_index] = np.minimum(ref_offset, estimated_intervals[:, 1]) > np.maximum(
            ref_onset, estimated_intervals[:, 0]
        )
    split = np.sum(overlap.sum(axis=1) > 1) / denominator
    merged = np.sum(overlap.sum(axis=0) > 1) / denominator
    spurious = np.sum(overlap.sum(axis=0) == 0) / denominator
    return float(split), float(merged), float(spurious)


def evaluate_notes(reference: pd.DataFrame, estimate: pd.DataFrame) -> dict[str, float | int]:
    reference_intervals, reference_hz = _intervals_and_hz(reference)
    estimated_intervals, estimated_hz = _intervals_and_hz(estimate)

    if len(reference_intervals) and len(estimated_intervals):
        cpo_p, cpo_r, cpo_f, _ = mir_eval.transcription.precision_recall_f1_overlap(
            reference_intervals,
            reference_hz,
            estimated_intervals,
            estimated_hz,
            onset_tolerance=ONSET_TOLERANCE,
            pitch_tolerance=PITCH_TOLERANCE_CENTS,
            offset_ratio=OFFSET_RATIO,
            offset_min_tolerance=OFFSET_MIN_TOLERANCE,
        )
        cp_p, cp_r, cp_f, _ = mir_eval.transcription.precision_recall_f1_overlap(
            reference_intervals,
            reference_hz,
            estimated_intervals,
            estimated_hz,
            onset_tolerance=ONSET_TOLERANCE,
            pitch_tolerance=PITCH_TOLERANCE_CENTS,
            offset_ratio=None,
        )
        con_p, con_r, con_f = mir_eval.transcription.onset_precision_recall_f1(
            reference_intervals,
            estimated_intervals,
            onset_tolerance=ONSET_TOLERANCE,
        )
    else:
        cpo_p = cpo_r = cpo_f = 0.0
        cp_p = cp_r = cp_f = 0.0
        con_p = con_r = con_f = 0.0

    split, merged, spurious = segmentation_error_rates(
        reference_intervals, estimated_intervals
    )
    return {
        "ref_notes": int(len(reference_intervals)),
        "est_notes": int(len(estimated_intervals)),
        "COnPOff_P": float(cpo_p),
        "COnPOff_R": float(cpo_r),
        "COnPOff_F": float(cpo_f),
        "COnP_P": float(cp_p),
        "COnP_R": float(cp_r),
        "COnP_F": float(cp_f),
        "COn_P": float(con_p),
        "COn_R": float(con_r),
        "COn_F": float(con_f),
        "Split": split,
        "Merged": merged,
        "Spurious": spurious,
    }


def conpoff_match_counts(
    reference: pd.DataFrame, estimate: pd.DataFrame
) -> tuple[int, int, int]:
    """Return COnPOff matches, reference notes, and estimated notes."""

    reference_intervals, reference_hz = _intervals_and_hz(reference)
    estimated_intervals, estimated_hz = _intervals_and_hz(estimate)
    if not len(reference_intervals) or not len(estimated_intervals):
        return 0, len(reference_intervals), len(estimated_intervals)
    matches = mir_eval.transcription.match_notes(
        reference_intervals,
        reference_hz,
        estimated_intervals,
        estimated_hz,
        onset_tolerance=ONSET_TOLERANCE,
        pitch_tolerance=PITCH_TOLERANCE_CENTS,
        offset_ratio=OFFSET_RATIO,
        offset_min_tolerance=OFFSET_MIN_TOLERANCE,
    )
    return len(matches), len(reference_intervals), len(estimated_intervals)


def f_measure_from_counts(matches: int, references: int, estimates: int) -> float:
    precision = matches / estimates if estimates else 0.0
    recall = matches / references if references else 0.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def concatenate_recordings(
    notes_by_stem: Mapping[str, pd.DataFrame], stems: list[str], gap_seconds: float = 10.0
) -> pd.DataFrame:
    shifted: list[pd.DataFrame] = []
    shift = 0.0
    for stem in stems:
        notes = notes_by_stem[stem].copy()
        if not notes.empty:
            notes[["onset", "offset"]] += shift
            shifted.append(notes)
            shift = float(notes["offset"].max()) + gap_seconds
        else:
            shift += gap_seconds
    if not shifted:
        return pd.DataFrame(columns=["onset", "offset", "midi", "hz"])
    return pd.concat(shifted, ignore_index=True)


def evaluate_corpus(
    reference_by_stem: Mapping[str, pd.DataFrame],
    estimate_by_stem: Mapping[str, pd.DataFrame],
    stems: list[str] | None = None,
) -> tuple[dict[str, float | int], pd.DataFrame, dict[str, float]]:
    ordered_stems = stems or sorted(reference_by_stem)
    per_recording_rows = []
    for stem in ordered_stems:
        metrics = evaluate_notes(reference_by_stem[stem], estimate_by_stem[stem])
        per_recording_rows.append({"stem": stem, **metrics})
    per_recording = pd.DataFrame(per_recording_rows)

    # Use the same shifts for references and estimates so recordings cannot
    # overlap after concatenation.
    reference_parts: list[pd.DataFrame] = []
    estimate_parts: list[pd.DataFrame] = []
    shift = 0.0
    for stem in ordered_stems:
        reference = reference_by_stem[stem].copy()
        estimate = estimate_by_stem[stem].copy()
        local_max = max(
            float(reference["offset"].max()) if not reference.empty else 0.0,
            float(estimate["offset"].max()) if not estimate.empty else 0.0,
        )
        reference[["onset", "offset"]] += shift
        estimate[["onset", "offset"]] += shift
        reference_parts.append(reference)
        estimate_parts.append(estimate)
        shift += local_max + 10.0
    reference_all = pd.concat(reference_parts, ignore_index=True)
    estimate_all = pd.concat(estimate_parts, ignore_index=True)
    micro = evaluate_notes(reference_all, estimate_all)

    macro_columns = ["COnPOff_F", "COnP_F", "COn_F", "Split", "Merged", "Spurious"]
    macro = {column: float(per_recording[column].mean()) for column in macro_columns}
    return micro, per_recording, macro
