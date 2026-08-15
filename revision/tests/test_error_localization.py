from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revision" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jasm_revision.error_localization import (
    boundary_window_coverage,
    phrase_boundaries,
    segmentation_error_events,
)


def test_phrase_boundaries_use_long_gap_midpoint() -> None:
    reference = pd.DataFrame(
        {
            "onset": [0.0, 0.5, 1.5],
            "offset": [0.4, 0.9, 2.0],
            "midi": [60.0, 62.0, 64.0],
        }
    )
    assert phrase_boundaries(reference, 0.3).tolist() == pytest.approx(
        [0.0, 0.9, 1.5, 2.0]
    )


def test_segmentation_events_are_located() -> None:
    reference = pd.DataFrame(
        {
            "onset": [0.0, 1.0, 2.0],
            "offset": [0.9, 1.9, 2.9],
            "midi": [60.0, 62.0, 64.0],
        }
    )
    estimate = pd.DataFrame(
        {
            "onset": [0.0, 0.5, 1.0, 3.2],
            "offset": [0.4, 1.4, 2.5, 3.5],
            "midi": [60.0, 60.0, 62.0, 65.0],
        }
    )
    events = segmentation_error_events("x", reference, estimate)
    assert (events["error_type"] == "split").sum() == 2
    assert (events["error_type"] == "merged").sum() == 2
    assert (events["error_type"] == "spurious").sum() == 1


def test_boundary_window_coverage_merges_overlaps() -> None:
    coverage = boundary_window_coverage(0.0, 1.0, [0.1, 0.2, 0.9], 0.15)
    assert coverage == pytest.approx(0.60)
