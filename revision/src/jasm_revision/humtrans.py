"""HumTrans MIDI readers and evaluation protocols.

The HumTrans repository uses a non-standard onset-only, octave-invariant
metric.  We reproduce that protocol separately from the standard Molina-style
metrics used elsewhere in this revision so the two cannot be confused.
"""

from __future__ import annotations

from pathlib import Path

import mir_eval
import numpy as np
import pandas as pd
import pretty_midi

from .evaluation import evaluate_notes, midi_to_hz


HUMTRANS_ONSET_TOLERANCE = 0.050
HUMTRANS_PITCH_TOLERANCE_CENTS = 1.0
HUMTRANS_OCTAVE_RADIUS = 16


def midi_file_to_notes(path: str | Path) -> pd.DataFrame:
    """Read all non-drum MIDI notes into the revision's canonical schema."""

    midi = pretty_midi.PrettyMIDI(str(path))
    rows: list[dict[str, float]] = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            if note.end <= note.start:
                continue
            rows.append(
                {
                    "onset": float(note.start),
                    "offset": float(note.end),
                    "midi": float(note.pitch),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["onset", "offset", "midi", "hz"])
    notes = pd.DataFrame(rows).sort_values(
        ["onset", "offset", "midi"], kind="stable"
    ).reset_index(drop=True)
    notes["hz"] = midi_to_hz(notes["midi"])
    return notes


def trim_estimates_to_reference_span(
    reference: pd.DataFrame, estimate: pd.DataFrame
) -> pd.DataFrame:
    """Apply the estimate trimming used by the official HumTrans script."""

    if reference.empty or estimate.empty:
        return estimate.iloc[0:0].copy() if reference.empty else estimate.copy()
    start = float(reference["onset"].min())
    end = float(reference["offset"].max())
    return estimate.loc[
        (estimate["onset"] >= start) & (estimate["onset"] <= end)
    ].reset_index(drop=True)


def _official_onset_metrics(
    reference: pd.DataFrame, estimate: pd.DataFrame, octave_shift: int
) -> tuple[float, float, float]:
    if reference.empty or estimate.empty:
        return 0.0, 0.0, 0.0
    reference_intervals = reference[["onset", "offset"]].to_numpy(np.float64)
    estimated_intervals = estimate[["onset", "offset"]].to_numpy(np.float64)
    reference_hz = midi_to_hz(
        reference["midi"].to_numpy(np.float64) + 12.0 * octave_shift
    )
    estimated_hz = midi_to_hz(estimate["midi"].to_numpy(np.float64))
    precision, recall, f_measure, _ = (
        mir_eval.transcription.precision_recall_f1_overlap(
            reference_intervals,
            reference_hz,
            estimated_intervals,
            estimated_hz,
            onset_tolerance=HUMTRANS_ONSET_TOLERANCE,
            pitch_tolerance=HUMTRANS_PITCH_TOLERANCE_CENTS,
            offset_ratio=None,
        )
    )
    return float(precision), float(recall), float(f_measure)


def evaluate_humtrans_official(
    reference: pd.DataFrame, estimate: pd.DataFrame
) -> dict[str, float | int]:
    """Reproduce the official onset-only, octave-invariant HumTrans metric."""

    trimmed = trim_estimates_to_reference_span(reference, estimate)
    candidates = [
        _official_onset_metrics(reference, trimmed, octave_shift)
        for octave_shift in range(-HUMTRANS_OCTAVE_RADIUS, HUMTRANS_OCTAVE_RADIUS + 1)
    ]
    best_shift, best = max(
        zip(
            range(-HUMTRANS_OCTAVE_RADIUS, HUMTRANS_OCTAVE_RADIUS + 1),
            candidates,
            strict=True,
        ),
        key=lambda item: item[1][2],
    )
    precision, recall, f_measure = best
    return {
        "official_precision": precision,
        "official_recall": recall,
        "official_f1": f_measure,
        "official_octave_shift": int(best_shift),
        "ref_notes": int(len(reference)),
        "est_notes_before_trim": int(len(estimate)),
        "est_notes_after_trim": int(len(trimmed)),
    }


def evaluate_humtrans_recording(
    reference: pd.DataFrame, estimate: pd.DataFrame
) -> dict[str, float | int]:
    """Return official HumTrans and standard Molina-style metrics together."""

    official = evaluate_humtrans_official(reference, estimate)
    trimmed = trim_estimates_to_reference_span(reference, estimate)
    standard = evaluate_notes(reference, trimmed)
    return {**official, **{f"standard_{key}": value for key, value in standard.items()}}
