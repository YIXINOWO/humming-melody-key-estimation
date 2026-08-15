from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revision" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jasm_revision.labels import make_boundary_labels


def test_boundary_label_radius() -> None:
    frames = pd.DataFrame(
        {"stem": ["x"] * 6, "time": [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]}
    )
    notes = {
        "x": pd.DataFrame({"onset": [0.02], "offset": [0.04], "midi": [60.0]})
    }
    labels = make_boundary_labels(frames, notes, radius_frames=1)
    assert labels["onset_label"].tolist() == [0, 1, 1, 1, 0, 0]
    assert labels["offset_label"].tolist() == [0, 0, 0, 1, 1, 1]

