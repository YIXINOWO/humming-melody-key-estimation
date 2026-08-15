#!/usr/bin/env python3
"""Check waveform feature extraction against the frozen historical cache."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revision" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jasm_revision.features import extract_features_from_wav  # noqa: E402


def main() -> int:
    config = yaml.safe_load(
        (ROOT / "revision" / "config" / "main_experiment.yaml").read_text(
            encoding="utf-8"
        )
    )
    historical = pd.read_csv(ROOT / config["data"]["historical_frame_table"])
    output_dir = ROOT / "revision" / "results" / "feature_reproduction"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for stem in ("child1", "q80"):
        reproduced = extract_features_from_wav(
            ROOT / "audio" / f"{stem}.wav",
            stem,
            lag_sources=config["features"]["lag_sources"],
            lags=config["features"]["lags"],
        )
        frozen = historical.loc[historical["stem"] == stem].reset_index(drop=True)
        if len(reproduced) != len(frozen):
            raise AssertionError(f"{stem}: {len(reproduced)} != {len(frozen)} frames")
        for column in config["features"]["columns"] + [
            f"{source}_lag{int(lag):+d}"
            for source in config["features"]["lag_sources"]
            for lag in config["features"]["lags"]
        ]:
            left = reproduced[column].to_numpy(np.float64)
            right = frozen[column].to_numpy(np.float64)
            valid = np.isfinite(left) & np.isfinite(right)
            rows.append(
                {
                    "stem": stem,
                    "feature": column,
                    "finite_values": int(valid.sum()),
                    "mae": float(np.mean(np.abs(left[valid] - right[valid]))),
                    "max_abs_error": float(np.max(np.abs(left[valid] - right[valid]))),
                }
            )
    audit = pd.DataFrame(rows)
    audit.to_csv(output_dir / "feature_reproduction.csv", index=False)
    summary = {
        "stems": ["child1", "q80"],
        "features": len(audit["feature"].unique()),
        "max_feature_mae": float(audit["mae"].max()),
        "max_feature_abs_error": float(audit["max_abs_error"].max()),
        "confidence_mae": float(
            audit.loc[audit["feature"] == "conf", "mae"].max()
        ),
        "pitch_mae": float(audit.loc[audit["feature"] == "midi", "mae"].max()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
