from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revision" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jasm_revision.features import build_frame_features, hz_to_midi


def test_hz_to_midi_maps_a4() -> None:
    assert hz_to_midi(np.asarray([440.0]))[0] == pytest.approx(69.0)


def test_build_frame_features_has_expected_lags() -> None:
    sample_rate = 16000
    waveform = np.zeros(sample_rate // 10, dtype=np.float32)
    track = pd.DataFrame(
        {
            "time": np.arange(10) * 0.01,
            "f0_hz": np.full(10, 440.0),
            "confidence": np.linspace(0.1, 1.0, 10),
            "midi": np.full(10, 69.0),
        }
    )
    features = build_frame_features(
        "x",
        waveform,
        sample_rate,
        track,
        lag_sources=["midi", "conf"],
        lags=[-1, 1],
    )
    assert features["midi"].tolist() == pytest.approx([69.0] * 10)
    assert features["conf_lag-1"].iloc[0] == pytest.approx(track["confidence"].iloc[1])
    assert features["conf_lag+1"].iloc[1] == pytest.approx(track["confidence"].iloc[0])
