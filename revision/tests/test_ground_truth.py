from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revision" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jasm_revision.ground_truth import GroundTruthFormatError, read_ground_truth


def test_note_rows_layout(tmp_path: Path) -> None:
    source = tmp_path / "rows.txt"
    source.write_text("0.10 0.40 60.0\n0.50 0.80 62.0\n", encoding="utf-8")
    record = read_ground_truth(source)
    assert record.layout == "note_rows"
    assert len(record.notes) == 2
    assert record.numeric_tokens == 6


def test_value_vector_layout(tmp_path: Path) -> None:
    source = tmp_path / "vector.txt"
    source.write_text("0.10\n0.40\n60.0\n0.50\n0.80\n62.0\n", encoding="utf-8")
    record = read_ground_truth(source)
    assert record.layout == "value_vector"
    assert len(record.notes) == 2
    assert record.notes.iloc[1]["midi"] == pytest.approx(62.0)


def test_incomplete_vector_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "broken.txt"
    source.write_text("0.10\n0.40\n60.0\n0.50\n", encoding="utf-8")
    with pytest.raises(GroundTruthFormatError, match="not divisible by 3"):
        read_ground_truth(source)


def test_invalid_interval_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "invalid.txt"
    source.write_text("0.40 0.10 60.0\n", encoding="utf-8")
    with pytest.raises(GroundTruthFormatError, match="offset <= onset"):
        read_ground_truth(source)


def test_real_corpus_counts_and_layouts() -> None:
    gt_dir = ROOT / "gt_files_temp"
    if not gt_dir.is_dir():
        pytest.skip("MTG-QBH/Molina annotations are not redistributed")
    records = [read_ground_truth(path) for path in sorted(gt_dir.glob("*.GroundTruth.txt"))]
    assert len(records) == 38
    assert sum(len(record.notes) for record in records) == 2152
    vector_counts = {
        record.path.name: len(record.notes)
        for record in records
        if record.layout == "value_vector"
    }
    assert vector_counts == {
        "q80.GroundTruth.txt": 56,
        "q85.GroundTruth.txt": 23,
        "q86.GroundTruth.txt": 51,
        "q87.GroundTruth.txt": 67,
    }
