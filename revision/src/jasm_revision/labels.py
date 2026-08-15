"""Boundary-label generation from audited note annotations."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def make_boundary_labels(
    frame_table: pd.DataFrame,
    notes_by_stem: Mapping[str, pd.DataFrame],
    radius_frames: int = 1,
) -> pd.DataFrame:
    """Return corrected onset/offset labels aligned to each recording's frames."""

    output = frame_table[["stem", "time"]].copy()
    output["onset_label"] = np.int8(0)
    output["offset_label"] = np.int8(0)
    for stem, frame_indices in frame_table.groupby("stem", sort=False).groups.items():
        if stem not in notes_by_stem:
            raise KeyError(f"missing annotations for {stem}")
        index_array = np.asarray(frame_indices, dtype=np.int64)
        times = frame_table.loc[index_array, "time"].to_numpy(dtype=np.float64)
        notes = notes_by_stem[stem]
        for column, label_column in (("onset", "onset_label"), ("offset", "offset_label")):
            labels = np.zeros(len(index_array), dtype=np.int8)
            for event_time in notes[column].to_numpy(dtype=np.float64):
                center = int(np.argmin(np.abs(times - event_time)))
                lower = max(0, center - radius_frames)
                upper = min(len(labels), center + radius_frames + 1)
                labels[lower:upper] = 1
            output.loc[index_array, label_column] = labels
    return output

