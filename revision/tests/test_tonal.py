from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revision" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jasm_revision.tonal import (
    audio_profile,
    profile_agreement,
    symbolic_profile,
    template_label,
)


def test_identical_representations_have_perfect_profile_agreement() -> None:
    profile = np.asarray([0.5, 0.0, 0.5] + [0.0] * 9)
    result = profile_agreement(profile, profile)
    assert result["profile_cosine_similarity"] == pytest.approx(1.0)
    assert result["jensen_shannon_distance"] == pytest.approx(0.0)
    assert result["dominant_pitch_class_match"]


def test_symbolic_profile_is_duration_weighted() -> None:
    notes = pd.DataFrame(
        {"onset": [0.0, 1.0], "offset": [1.0, 4.0], "midi": [60.0, 62.0]}
    )
    profile = symbolic_profile(notes)
    assert profile[0] == pytest.approx(0.25)
    assert profile[2] == pytest.approx(0.75)


def test_audio_profile_applies_confidence_threshold() -> None:
    frames = pd.DataFrame(
        {
            "midi": [60.0, 62.0],
            "conf": [0.2, 0.8],
            "voiced": [1.0, 1.0],
        }
    )
    profile = audio_profile(frames, 0.3)
    assert profile[0] == pytest.approx(0.0)
    assert profile[2] == pytest.approx(1.0)


def test_template_label_names_tonic_and_mode() -> None:
    profile = np.zeros(12)
    profile[[0, 4, 7]] = [0.5, 0.25, 0.25]
    label = template_label(profile)
    assert isinstance(label["tonic"], int)
    assert label["mode"] in {"major", "minor"}
    assert label["label"].endswith(str(label["mode"]))
