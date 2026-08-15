from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revision" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jasm_revision.evaluation import evaluate_notes


def test_identical_notes_score_one() -> None:
    notes = pd.DataFrame(
        {"onset": [0.1, 0.5], "offset": [0.4, 0.9], "midi": [60.0, 62.0]}
    )
    result = evaluate_notes(notes, notes.copy())
    assert result["COnPOff_F"] == pytest.approx(1.0)
    assert result["COnP_F"] == pytest.approx(1.0)
    assert result["COn_F"] == pytest.approx(1.0)
    assert result["Spurious"] == pytest.approx(0.0)
