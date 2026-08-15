#!/usr/bin/env python3
"""Fit a Molina-only frozen model for untouched external-corpus evaluation."""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import joblib
import mir_eval
import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revision" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jasm_revision.ground_truth import read_ground_truth  # noqa: E402
from jasm_revision.labels import make_boundary_labels  # noqa: E402
from jasm_revision.nested_cv import (  # noqa: E402
    _candidate_parameters,
    _decoder_candidates,
    _fit_probability_model,
    _select_decoder,
    feature_columns,
    grouped_splits,
)


def main() -> int:
    started = time.time()
    config_path = ROOT / "revision" / "config" / "main_experiment.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["project"]["seed"])
    output_dir = ROOT / "revision" / "results" / "external_deployment_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    references = {
        path.name.removesuffix(".GroundTruth.txt"): read_ground_truth(path).notes
        for path in sorted((ROOT / config["data"]["ground_truth_dir"]).glob("*.GroundTruth.txt"))
    }
    frames = pd.read_csv(ROOT / config["data"]["historical_frame_table"])
    frames.index.name = "row_index"
    columns = feature_columns(config, frames.columns.tolist())
    x = (
        frames[columns]
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
        .to_numpy(np.float32)
    )
    labels = make_boundary_labels(frames, references, radius_frames=1)
    y_onset = labels["onset_label"].to_numpy(np.int8)
    y_offset = labels["offset_label"].to_numpy(np.int8)
    groups = frames["stem"].to_numpy()
    splits = grouped_splits(groups, int(config["evaluation"]["outer_splits"]), seed + 7000)

    model_candidates = _candidate_parameters(config, smoke_test=False)
    candidate_probabilities = []
    model_rows = []
    for candidate_index, parameters in enumerate(model_candidates):
        onset_probability = np.zeros(len(frames), dtype=np.float64)
        offset_probability = np.zeros(len(frames), dtype=np.float64)
        for fold_index, (train, validation) in enumerate(splits, start=1):
            onset_probability[validation] = _fit_probability_model(
                x[train],
                y_onset[train],
                x[validation],
                parameters,
                seed + 710000 + candidate_index * 100 + fold_index,
            )
            offset_probability[validation] = _fit_probability_model(
                x[train],
                y_offset[train],
                x[validation],
                parameters,
                seed + 720000 + candidate_index * 100 + fold_index,
            )
        onset_ap = float(average_precision_score(y_onset, onset_probability))
        offset_ap = float(average_precision_score(y_offset, offset_probability))
        model_rows.append(
            {
                "model_candidate": candidate_index,
                **parameters,
                "onset_average_precision": onset_ap,
                "offset_average_precision": offset_ap,
                "mean_frame_average_precision": (onset_ap + offset_ap) / 2.0,
            }
        )
        candidate_probabilities.append(
            pd.DataFrame(
                {
                    "row_index": frames.index,
                    "onset_probability": onset_probability,
                    "offset_probability": offset_probability,
                }
            )
        )
        print(
            f"candidate {candidate_index}: onset_AP={onset_ap:.4f}, "
            f"offset_AP={offset_ap:.4f}",
            flush=True,
        )

    model_search = pd.DataFrame(model_rows).sort_values(
        ["mean_frame_average_precision", "model_candidate"],
        ascending=[False, True],
    )
    model_search.to_csv(output_dir / "model_selection.csv", index=False)
    selected_index = int(model_search.iloc[0]["model_candidate"])
    selected_parameters = model_candidates[selected_index]
    selected_probabilities = candidate_probabilities[selected_index]

    decoder, decoder_search = _select_decoder(
        frames,
        selected_probabilities,
        references,
        sorted(references),
        _decoder_candidates(config, smoke_test=False),
    )
    decoder_search.to_csv(output_dir / "decoder_selection.csv", index=False)
    selected_probabilities.to_csv(
        output_dir / "molina_oof_probabilities.csv.gz",
        index=False,
        compression="gzip",
    )

    onset_model = RandomForestClassifier(**selected_parameters, random_state=seed + 730001)
    offset_model = RandomForestClassifier(**selected_parameters, random_state=seed + 730002)
    onset_model.fit(x, y_onset)
    offset_model.fit(x, y_offset)
    joblib.dump(onset_model, output_dir / "onset_model.joblib", compress=3)
    joblib.dump(offset_model, output_dir / "offset_model.joblib", compress=3)

    manifest = {
        "purpose": "frozen Molina-trained deployment model for external evaluation only",
        "training_recordings": len(references),
        "training_reference_notes": int(sum(len(notes) for notes in references.values())),
        "selection_protocol": (
            "5-fold recording-grouped OOF on Molina only; model chosen by mean "
            "onset/offset frame AP and decoder chosen by OOF COnPOff"
        ),
        "external_test_involvement": "none",
        "selected_model_candidate": selected_index,
        "model_parameters": selected_parameters,
        "decoder_parameters": decoder,
        "feature_columns": columns,
        "feature_extractor": {
            "crepe_variant": "full",
            "sample_rate": 16000,
            "frame_hop_seconds": 0.01,
            "fmin_hz": 65.0,
            "fmax_hz": 1000.0,
            "crepe_median_filter": 9,
            "pitch_median_filter": 5,
        },
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "mir_eval": mir_eval.__version__,
        },
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
