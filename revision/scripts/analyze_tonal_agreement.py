#!/usr/bin/env python3
"""Replace circular key accuracy with representation-agreement analysis."""

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

from jasm_revision.ground_truth import read_ground_truth  # noqa: E402
from jasm_revision.tonal import (  # noqa: E402
    audio_profile,
    profile_agreement,
    symbolic_profile,
    template_label,
)


PRIMARY_CONFIDENCE_THRESHOLD = 0.30
BOOTSTRAP_ITERATIONS = 10000
SEED = 2026


def bootstrap_summary(per_recording: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for metric in ("profile_cosine_similarity", "jensen_shannon_distance"):
        values = per_recording[metric].to_numpy(np.float64)
        sampled = values[
            rng.integers(0, len(values), size=(BOOTSTRAP_ITERATIONS, len(values)))
        ].mean(axis=1)
        rows.append(
            {
                "metric": metric,
                "macro_mean": float(values.mean()),
                "bootstrap_ci_low": float(np.percentile(sampled, 2.5)),
                "bootstrap_ci_high": float(np.percentile(sampled, 97.5)),
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            }
        )
    return pd.DataFrame(rows)


def analyze_threshold(
    references: dict[str, pd.DataFrame], frames: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    rows = []
    for stem in sorted(references):
        reference_profile = symbolic_profile(references[stem])
        estimate_profile = audio_profile(
            frames.loc[frames["stem"] == stem], threshold
        )
        reference_label = template_label(reference_profile)
        estimate_label = template_label(estimate_profile)
        rows.append(
            {
                "stem": stem,
                "confidence_threshold": threshold,
                **profile_agreement(reference_profile, estimate_profile),
                "symbolic_dominant_pitch_class": int(np.argmax(reference_profile)),
                "audio_dominant_pitch_class": int(np.argmax(estimate_profile)),
                "symbolic_template_label": reference_label["label"],
                "audio_template_label": estimate_label["label"],
                "descriptive_exact_label_agreement": (
                    reference_label["label"] == estimate_label["label"]
                ),
                "descriptive_tonic_agreement": (
                    reference_label["tonic"] == estimate_label["tonic"]
                ),
                "descriptive_mode_agreement": (
                    reference_label["mode"] == estimate_label["mode"]
                ),
                **{
                    f"symbolic_pc_{index}": value
                    for index, value in enumerate(reference_profile)
                },
                **{
                    f"audio_pc_{index}": value
                    for index, value in enumerate(estimate_profile)
                },
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    output_dir = ROOT / "revision" / "results" / "molina_tonal_agreement"
    output_dir.mkdir(parents=True, exist_ok=True)
    references = {
        path.name.removesuffix(".GroundTruth.txt"): read_ground_truth(path).notes
        for path in sorted((ROOT / "gt_files_temp").glob("*.GroundTruth.txt"))
    }
    config = yaml.safe_load(
        (ROOT / "revision" / "config" / "main_experiment.yaml").read_text(
            encoding="utf-8"
        )
    )
    frames = pd.read_csv(
        ROOT / config["data"]["historical_frame_table"],
        usecols=["stem", "midi", "conf", "voiced"],
    )
    sensitivity_parts = [
        analyze_threshold(references, frames, threshold)
        for threshold in (0.20, 0.25, 0.30)
    ]
    sensitivity = pd.concat(sensitivity_parts, ignore_index=True)
    primary = sensitivity.loc[
        sensitivity["confidence_threshold"] == PRIMARY_CONFIDENCE_THRESHOLD
    ].reset_index(drop=True)
    primary.to_csv(output_dir / "per_recording_agreement.csv", index=False)
    sensitivity.to_csv(output_dir / "confidence_sensitivity.csv", index=False)
    bootstrap = bootstrap_summary(primary)
    bootstrap.to_csv(output_dir / "recording_bootstrap_ci.csv", index=False)

    label_metrics = {
        column: float(primary[column].mean())
        for column in (
            "dominant_pitch_class_match",
            "descriptive_exact_label_agreement",
            "descriptive_tonic_agreement",
            "descriptive_mode_agreement",
        )
    }
    sensitivity_summary = (
        sensitivity.groupby("confidence_threshold")
        .agg(
            profile_cosine_similarity=("profile_cosine_similarity", "mean"),
            jensen_shannon_distance=("jensen_shannon_distance", "mean"),
            dominant_pitch_class_match=("dominant_pitch_class_match", "mean"),
            descriptive_exact_label_agreement=(
                "descriptive_exact_label_agreement",
                "mean",
            ),
        )
        .reset_index()
    )
    summary = {
        "analysis_type": "tonal_representation_agreement_not_key_accuracy",
        "recordings": len(primary),
        "primary_confidence_threshold": PRIMARY_CONFIDENCE_THRESHOLD,
        "primary_macro_metrics": {
            "profile_cosine_similarity": float(
                primary["profile_cosine_similarity"].mean()
            ),
            "jensen_shannon_distance": float(
                primary["jensen_shannon_distance"].mean()
            ),
            **label_metrics,
        },
        "bootstrap": bootstrap.to_dict(orient="records"),
        "confidence_sensitivity": sensitivity_summary.to_dict(orient="records"),
        "interpretation_guardrail": (
            "Template-derived label matches are descriptive agreement between two "
            "representations. They are not key-prediction accuracy because no "
            "independently annotated key labels are available."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Molina tonal-representation agreement",
        "",
        summary["interpretation_guardrail"],
        "",
        f"Primary CREPE confidence threshold: {PRIMARY_CONFIDENCE_THRESHOLD:.2f}.",
        "",
        "| Measure | Macro value |",
        "|---|---:|",
        f"| Pitch-class profile cosine similarity | {summary['primary_macro_metrics']['profile_cosine_similarity']:.4f} |",
        f"| Jensen-Shannon distance | {summary['primary_macro_metrics']['jensen_shannon_distance']:.4f} |",
        f"| Dominant pitch-class agreement | {summary['primary_macro_metrics']['dominant_pitch_class_match']:.4f} |",
        f"| Descriptive exact template-label agreement | {summary['primary_macro_metrics']['descriptive_exact_label_agreement']:.4f} |",
        f"| Descriptive tonic agreement | {summary['primary_macro_metrics']['descriptive_tonic_agreement']:.4f} |",
        f"| Descriptive mode agreement | {summary['primary_macro_metrics']['descriptive_mode_agreement']:.4f} |",
        "",
        "The first three measures compare the two pitch-class representations "
        "directly and do not use Krumhansl-Schmuckler as the evaluator. Confidence "
        "threshold sensitivity (0.20/0.25/0.30) and recording-bootstrap intervals "
        "are provided as separate artifacts.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
