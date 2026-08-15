#!/usr/bin/env python3
"""Exploratory corrected-label feature ablation with recording-grouped OOF fits."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revision" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jasm_revision.decoder import decode_notes  # noqa: E402
from jasm_revision.evaluation import evaluate_corpus  # noqa: E402
from jasm_revision.ground_truth import read_ground_truth  # noqa: E402
from jasm_revision.labels import make_boundary_labels  # noqa: E402


SEED = 2026
DECODER = {
    "onset_threshold": 0.36,
    "offset_threshold": 0.405,
    "min_onset_separation_seconds": 0.162,
    "min_note_duration_seconds": 0.10,
    "crepe_confidence_threshold": 0.30,
}


def feature_sets(config: dict) -> list[tuple[str, list[str]]]:
    base = list(config["features"]["columns"])
    lags = [
        f"{source}_lag{int(lag):+d}"
        for source in config["features"]["lag_sources"]
        for lag in config["features"]["lags"]
    ]
    f0 = ["midi", "conf", "voiced", "dm", "abs_dm", "ddm"]
    energy = ["rms", "drms"]
    flux = ["flux", "dflux"]
    context = ["local_pitch_std", "voice_since_start", "voice_to_end", "voice_run_pos"]
    return [
        ("F0 only", f0),
        ("+ RMS energy", f0 + energy),
        ("+ spectral flux", f0 + energy + flux),
        ("+ voicing-run / pitch stability", f0 + energy + flux + context),
        ("+ temporal lags (full)", base + lags),
    ]


def fit_probability(x_train, y_train, x_test, seed):
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=6,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    classes = model.classes_.tolist()
    return model.predict_proba(x_test)[:, classes.index(1)] if 1 in classes else np.zeros(len(x_test))


def main() -> int:
    config = yaml.safe_load((ROOT / "revision" / "config" / "main_experiment.yaml").read_text(encoding="utf-8"))
    gt_dir = ROOT / config["data"]["ground_truth_dir"]
    references = {
        p.name.removesuffix(".GroundTruth.txt"): read_ground_truth(p).notes
        for p in sorted(gt_dir.glob("*.GroundTruth.txt"))
    }
    frame = pd.read_csv(ROOT / config["data"]["historical_frame_table"])
    frame.index.name = "row_index"
    labels = make_boundary_labels(frame, references, radius_frames=1)
    groups = frame["stem"].to_numpy()
    splits = list(GroupKFold(n_splits=5).split(frame, groups=groups))
    refs = references
    out = ROOT / "revision" / "results" / "corrected_ablation"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    predictions_by_config = {}
    for config_index, (name, columns) in enumerate(feature_sets(config)):
        x = frame[columns].replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(np.float32)
        onset = np.zeros(len(frame), dtype=np.float64)
        offset = np.zeros(len(frame), dtype=np.float64)
        for fold, (train, test) in enumerate(splits, start=1):
            onset[test] = fit_probability(x[train], labels["onset_label"].to_numpy()[train], x[test], SEED + config_index * 100 + fold)
            offset[test] = fit_probability(x[train], labels["offset_label"].to_numpy()[train], x[test], SEED + 5000 + config_index * 100 + fold)
        pred = {}
        for stem in sorted(references):
            part = frame.loc[frame["stem"] == stem].copy()
            part["onset_probability"] = onset[part.index]
            part["offset_probability"] = offset[part.index]
            pred[stem] = decode_notes(part, **DECODER)
        micro, per, macro = evaluate_corpus(refs, pred, sorted(refs))
        predictions_by_config[name] = pred
        rows.append({
            "configuration": name,
            "n_features": len(columns),
            "COnPOff_F_micro": micro["COnPOff_F"],
            "COnP_F_micro": micro["COnP_F"],
            "COn_F_micro": micro["COn_F"],
            "COnPOff_F_macro": macro["COnPOff_F"],
            "COnP_F_macro": macro["COnP_F"],
            "COn_F_macro": macro["COn_F"],
        })
        per.insert(0, "configuration", name)
        per.to_csv(out / f"{config_index+1:02d}_per_recording.csv", index=False)
        print(name, rows[-1], flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(out / "summary.csv", index=False)
    (out / "summary.json").write_text(json.dumps({"decoder": DECODER, "folds": 5, "rows": rows}, indent=2) + "\n", encoding="utf-8")

    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    xloc = np.arange(len(result))
    width = 0.24
    for delta, column, color, label in [(-width, "COnPOff_F_micro", "#3973A9", "COnPOff"), (0, "COnP_F_micro", "#47A6A0", "COnP"), (width, "COn_F_micro", "#6C7A89", "COn")]:
        values = result[column].to_numpy()
        ax.bar(xloc + delta, values, width, color=color, label=label)
        for i, value in enumerate(values):
            if column == "COnPOff_F_micro":
                ax.text(i + delta, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(xloc, ["F0", "+E", "+Flux", "+Context", "+Lags"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Micro F-measure")
    ax.set_xlabel("Cumulative feature configuration")
    ax.set_title("Exploratory corrected-label ablation", fontsize=8)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    for ext, kwargs in [("svg", {}), ("pdf", {}), ("tiff", {"dpi": 600}), ("png", {"dpi": 300})]:
        fig.savefig(out / f"fig_corrected_ablation.{ext}", bbox_inches="tight", **kwargs)
    fig.savefig(ROOT / "submission_asmp_latex_resubmission_source" / "fig_corrected_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
