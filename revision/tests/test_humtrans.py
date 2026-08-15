from pathlib import Path
import sys

import pandas as pd
import pretty_midi
import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revision" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jasm_revision.humtrans import (
    evaluate_humtrans_official,
    midi_file_to_notes,
    trim_estimates_to_reference_span,
)


def test_midi_file_to_notes(tmp_path: Path) -> None:
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.1, end=0.4))
    midi.instruments.append(instrument)
    path = tmp_path / "notes.mid"
    midi.write(str(path))
    notes = midi_file_to_notes(path)
    assert len(notes) == 1
    assert notes.iloc[0]["midi"] == pytest.approx(60.0)
    assert notes.iloc[0]["onset"] == pytest.approx(0.1, abs=0.002)


def test_official_metric_is_octave_invariant() -> None:
    reference = pd.DataFrame(
        {"onset": [0.1], "offset": [0.4], "midi": [60.0]}
    )
    estimate = pd.DataFrame(
        {"onset": [0.1], "offset": [0.3], "midi": [72.0]}
    )
    result = evaluate_humtrans_official(reference, estimate)
    assert result["official_f1"] == pytest.approx(1.0)
    assert abs(result["official_octave_shift"]) == 1


def test_estimates_are_trimmed_by_onset() -> None:
    reference = pd.DataFrame(
        {"onset": [1.0], "offset": [2.0], "midi": [60.0]}
    )
    estimate = pd.DataFrame(
        {
            "onset": [0.5, 1.1, 2.0, 2.1],
            "offset": [0.8, 1.4, 2.2, 2.4],
            "midi": [60.0, 60.0, 60.0, 60.0],
        }
    )
    trimmed = trim_estimates_to_reference_span(reference, estimate)
    assert trimmed["onset"].tolist() == [1.1, 2.0]
