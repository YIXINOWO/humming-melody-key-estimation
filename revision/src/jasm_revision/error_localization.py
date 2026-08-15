"""Operational phrase-boundary definitions and segmentation-error localization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def phrase_boundaries(
    reference: pd.DataFrame, gap_threshold_seconds: float = 0.30
) -> np.ndarray:
    """Return phrase starts/ends separated by sufficiently long inter-note gaps."""

    if reference.empty:
        return np.empty(0, dtype=np.float64)
    notes = reference.sort_values(["onset", "offset"], kind="stable").reset_index(
        drop=True
    )
    boundaries = [float(notes.iloc[0]["onset"]), float(notes.iloc[-1]["offset"])]
    for previous, following in zip(
        notes.iloc[:-1].itertuples(index=False),
        notes.iloc[1:].itertuples(index=False),
        strict=True,
    ):
        gap = float(following.onset - previous.offset)
        if gap >= gap_threshold_seconds:
            boundaries.extend([float(previous.offset), float(following.onset)])
    return np.unique(np.asarray(boundaries, dtype=np.float64))


def _overlap_matrix(reference: pd.DataFrame, estimate: pd.DataFrame) -> np.ndarray:
    if reference.empty or estimate.empty:
        return np.zeros((len(reference), len(estimate)), dtype=bool)
    ref_onset = reference["onset"].to_numpy(np.float64)[:, None]
    ref_offset = reference["offset"].to_numpy(np.float64)[:, None]
    est_onset = estimate["onset"].to_numpy(np.float64)[None, :]
    est_offset = estimate["offset"].to_numpy(np.float64)[None, :]
    return np.minimum(ref_offset, est_offset) > np.maximum(ref_onset, est_onset)


def segmentation_error_events(
    stem: str, reference: pd.DataFrame, estimate: pd.DataFrame
) -> pd.DataFrame:
    """Locate excess split boundaries, swallowed merge boundaries, and spurious notes."""

    reference = reference.sort_values(["onset", "offset"], kind="stable").reset_index(
        drop=True
    )
    estimate = estimate.sort_values(["onset", "offset"], kind="stable").reset_index(
        drop=True
    )
    overlap = _overlap_matrix(reference, estimate)
    rows: list[dict[str, float | int | str]] = []

    for ref_index in np.flatnonzero(overlap.sum(axis=1) > 1):
        estimate_indices = np.flatnonzero(overlap[ref_index])
        ordered = estimate_indices[
            np.argsort(estimate.iloc[estimate_indices]["onset"].to_numpy())
        ]
        for estimate_index in ordered[1:]:
            rows.append(
                {
                    "stem": stem,
                    "error_type": "split",
                    "event_time": float(estimate.iloc[estimate_index]["onset"]),
                    "reference_index": int(ref_index),
                    "estimate_index": int(estimate_index),
                }
            )

    for estimate_index in np.flatnonzero(overlap.sum(axis=0) > 1):
        reference_indices = np.flatnonzero(overlap[:, estimate_index])
        ordered = reference_indices[
            np.argsort(reference.iloc[reference_indices]["onset"].to_numpy())
        ]
        for reference_index in ordered[1:]:
            rows.append(
                {
                    "stem": stem,
                    "error_type": "merged",
                    "event_time": float(reference.iloc[reference_index]["onset"]),
                    "reference_index": int(reference_index),
                    "estimate_index": int(estimate_index),
                }
            )

    for estimate_index in np.flatnonzero(overlap.sum(axis=0) == 0):
        rows.append(
            {
                "stem": stem,
                "error_type": "spurious",
                "event_time": float(estimate.iloc[estimate_index]["onset"]),
                "reference_index": -1,
                "estimate_index": int(estimate_index),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "stem",
            "error_type",
            "event_time",
            "reference_index",
            "estimate_index",
        ],
    )


def distance_to_boundaries(times: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    if not len(times):
        return np.empty(0, dtype=np.float64)
    if not len(boundaries):
        return np.full(len(times), np.inf, dtype=np.float64)
    return np.min(np.abs(times[:, None] - boundaries[None, :]), axis=1)


def boundary_window_coverage(
    span_start: float, span_end: float, boundaries: np.ndarray, window_seconds: float
) -> float:
    """Return the fraction of the evaluation span covered by boundary windows."""

    duration = span_end - span_start
    if duration <= 0 or not len(boundaries):
        return 0.0
    intervals = sorted(
        (
            max(span_start, float(boundary - window_seconds)),
            min(span_end, float(boundary + window_seconds)),
        )
        for boundary in boundaries
    )
    covered = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            covered += current_end - current_start
            current_start, current_end = start, end
    covered += current_end - current_start
    return float(covered / duration)
