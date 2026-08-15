"""Deterministic conversion of onset/offset probabilities into note events."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import midi_to_hz


NOTE_COLUMNS = ["onset", "offset", "midi", "hz"]


def _contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, mask.astype(bool), False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(changes[index]), int(changes[index + 1])) for index in range(0, len(changes), 2)]


def _fill_short_gaps(mask: np.ndarray, maximum_gap_frames: int = 3) -> np.ndarray:
    output = mask.astype(bool).copy()
    for start, end in _contiguous_regions(~output):
        if (
            end - start <= maximum_gap_frames
            and start > 0
            and end < len(output)
            and output[start - 1]
            and output[end]
        ):
            output[start:end] = True
    return output


def _pick_peaks(
    times: np.ndarray, scores: np.ndarray, threshold: float, minimum_separation: float
) -> np.ndarray:
    candidates = [
        index
        for index in range(1, len(scores) - 1)
        if scores[index] >= threshold
        and scores[index] >= scores[index - 1]
        and scores[index] >= scores[index + 1]
    ]
    candidates.sort(key=lambda index: scores[index], reverse=True)
    selected: list[int] = []
    for index in candidates:
        if all(abs(times[index] - times[other]) >= minimum_separation for other in selected):
            selected.append(index)
    return np.asarray(sorted(selected), dtype=np.int64)


def decode_notes(
    frames: pd.DataFrame,
    *,
    onset_threshold: float,
    offset_threshold: float,
    min_onset_separation_seconds: float,
    min_note_duration_seconds: float,
    crepe_confidence_threshold: float,
) -> pd.DataFrame:
    times = frames["time"].to_numpy(dtype=np.float64)
    if len(times) < 2:
        return pd.DataFrame(columns=NOTE_COLUMNS)
    midi = frames["midi"].to_numpy(dtype=np.float64)
    confidence = frames["conf"].to_numpy(dtype=np.float64)
    voiced = (frames["voiced"].to_numpy(dtype=np.float64) > 0.5) & (
        confidence >= crepe_confidence_threshold
    )
    voiced = _fill_short_gaps(voiced, maximum_gap_frames=3)
    onset_probability = frames["onset_probability"].to_numpy(dtype=np.float64)
    offset_probability = frames["offset_probability"].to_numpy(dtype=np.float64)
    onset_indices = _pick_peaks(
        times, onset_probability, onset_threshold, min_onset_separation_seconds
    )
    offset_indices = _pick_peaks(
        times, offset_probability, offset_threshold, min_onset_separation_seconds * 0.6
    )
    hop = float(np.median(np.diff(times)))

    notes: list[dict[str, float]] = []
    for run_start, run_end in _contiguous_regions(voiced):
        run_onsets = [int(index) for index in onset_indices if run_start <= index < run_end]
        run_offsets = [int(index) for index in offset_indices if run_start < index <= run_end]
        if not run_onsets:
            run_onsets = [run_start]
        if run_onsets[0] - run_start > int(round(0.120 / hop)):
            run_onsets = [run_start, *run_onsets]
        run_onsets = sorted(set(run_onsets))

        for onset_number, onset_frame in enumerate(run_onsets):
            next_onset = (
                run_onsets[onset_number + 1]
                if onset_number + 1 < len(run_onsets)
                else run_end
            )
            minimum_offset = onset_frame + max(
                1, int(round(min_note_duration_seconds / hop))
            )
            maximum_offset = min(next_onset + int(round(0.120 / hop)), run_end)
            candidates = [
                index for index in run_offsets if minimum_offset <= index <= maximum_offset
            ]
            if candidates:
                before_next = [index for index in candidates if index <= next_onset]
                offset_frame = (
                    before_next[-1]
                    if before_next
                    else max(candidates, key=lambda index: offset_probability[index])
                )
            else:
                offset_frame = next_onset
            if offset_frame <= onset_frame:
                continue

            onset = float(times[min(onset_frame, len(times) - 1)])
            offset = float(times[min(offset_frame, len(times) - 1)] + hop)
            if offset - onset < min_note_duration_seconds:
                continue
            pitch_values = midi[onset_frame:offset_frame]
            pitch_values = pitch_values[np.isfinite(pitch_values)]
            if len(pitch_values) < 2:
                continue
            note_midi = float(np.median(pitch_values))
            notes.append(
                {
                    "onset": onset,
                    "offset": offset,
                    "midi": note_midi,
                    "hz": float(midi_to_hz(note_midi)),
                }
            )
    return pd.DataFrame(notes, columns=NOTE_COLUMNS)

