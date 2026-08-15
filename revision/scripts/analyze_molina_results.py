#!/usr/bin/env python3
"""Generate paper-ready secondary analyses from formal outer-test predictions."""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import mir_eval
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revision" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jasm_revision.decoder import decode_notes  # noqa: E402
from jasm_revision.evaluation import evaluate_corpus, midi_to_hz  # noqa: E402
from jasm_revision.ground_truth import read_ground_truth  # noqa: E402


def audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def load_references() -> dict[str, pd.DataFrame]:
    return {
        path.name.removesuffix(".GroundTruth.txt"): read_ground_truth(path).notes
        for path in sorted((ROOT / "gt_files_temp").glob("*.GroundTruth.txt"))
    }


def historical_frame_table() -> Path:
    config = yaml.safe_load(
        (ROOT / "revision" / "config" / "main_experiment.yaml").read_text(
            encoding="utf-8"
        )
    )
    return ROOT / config["data"]["historical_frame_table"]


def frame_level_metrics(
    frames: pd.DataFrame, references: dict[str, pd.DataFrame]
) -> tuple[dict[str, float], pd.DataFrame]:
    rows = []
    all_ref_time = []
    all_ref_frequency = []
    all_est_time = []
    all_est_frequency = []
    shift = 0.0
    for stem, part in frames.groupby("stem", sort=True):
        times = part["time"].to_numpy(dtype=np.float64)
        reference_frequency = np.zeros(len(times), dtype=np.float64)
        for note in references[stem].itertuples(index=False):
            mask = (times >= note.onset) & (times < note.offset)
            reference_frequency[mask] = midi_to_hz(note.midi)
        estimated_frequency = midi_to_hz(part["midi"].to_numpy(dtype=np.float64))
        estimated_voicing = part["conf"].to_numpy(dtype=np.float64)
        scores = mir_eval.melody.evaluate(
            times,
            reference_frequency,
            times,
            estimated_frequency,
            est_voicing=estimated_voicing,
            voicing_threshold=0.3,
        )
        rows.append({"stem": stem, **scores})
        all_ref_time.append(times + shift)
        all_est_time.append(times + shift)
        all_ref_frequency.append(reference_frequency)
        all_est_frequency.append(estimated_frequency)
        shift += float(times[-1]) + 10.0
    per_recording = pd.DataFrame(rows)
    micro = mir_eval.melody.evaluate(
        np.concatenate(all_ref_time),
        np.concatenate(all_ref_frequency),
        np.concatenate(all_est_time),
        np.concatenate(all_est_frequency),
        est_voicing=np.concatenate(
            [
                group["conf"].to_numpy(dtype=np.float64)
                for _, group in frames.groupby("stem", sort=True)
            ]
        ),
        voicing_threshold=0.3,
    )
    return dict(micro), per_recording


def parameter_robustness(
    frames: pd.DataFrame,
    probabilities: pd.DataFrame,
    references: dict[str, pd.DataFrame],
    selections: list[dict],
) -> pd.DataFrame:
    probability_lookup = probabilities.set_index("row_index")
    rows = []
    perturbations = [
        ("minus_10_percent", 0.9),
        ("selected", 1.0),
        ("plus_10_percent", 1.1),
    ]
    for label, factor in perturbations:
        estimate_by_stem: dict[str, pd.DataFrame] = {}
        for selection in selections:
            parameters = dict(selection["decoder_parameters"])
            for key in ("onset_threshold", "offset_threshold"):
                parameters[key] = float(np.clip(parameters[key] * factor, 0.0, 1.0))
            parameters["min_onset_separation_seconds"] *= factor
            parameters["min_note_duration_seconds"] *= factor
            for stem in selection["test_stems"]:
                part = frames.loc[frames["stem"] == stem].copy()
                part["onset_probability"] = probability_lookup.loc[
                    part.index, "onset_probability"
                ].to_numpy(dtype=np.float64)
                part["offset_probability"] = probability_lookup.loc[
                    part.index, "offset_probability"
                ].to_numpy(dtype=np.float64)
                estimate_by_stem[stem] = decode_notes(part, **parameters)
        micro, _, macro = evaluate_corpus(references, estimate_by_stem, sorted(references))
        rows.append(
            {
                "setting": label,
                "factor": factor,
                **{f"micro_{key}": value for key, value in micro.items()},
                **{f"macro_{key}": value for key, value in macro.items()},
            }
        )
    return pd.DataFrame(rows)


def bootstrap_macro(
    per_recording: pd.DataFrame, iterations: int = 10000, seed: int = 2026
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metrics = ["COnPOff_F", "COnP_F", "COn_F"]
    rows = []
    for metric in metrics:
        values = per_recording[metric].to_numpy(dtype=np.float64)
        sampled = values[rng.integers(0, len(values), size=(iterations, len(values)))].mean(
            axis=1
        )
        rows.append(
            {
                "metric": metric,
                "estimate": float(values.mean()),
                "ci_low": float(np.percentile(sampled, 2.5)),
                "ci_high": float(np.percentile(sampled, 97.5)),
                "iterations": iterations,
            }
        )
    return pd.DataFrame(rows)


def error_burden(per_recording: pd.DataFrame) -> pd.DataFrame:
    output = per_recording.copy()
    output["audio_duration_seconds"] = output["stem"].map(
        lambda stem: audio_duration(ROOT / "audio" / f"{stem}.wav")
    )
    for error in ("Split", "Merged", "Spurious"):
        output[f"{error.lower()}_approx_count"] = output[error] * output["ref_notes"]
        output[f"{error.lower()}_per_5_seconds"] = (
            output[f"{error.lower()}_approx_count"]
            / output["audio_duration_seconds"]
            * 5.0
        )
        output[f"{error.lower()}_per_100_ref_notes"] = output[error] * 100.0
    return output


def main() -> int:
    run_dir = ROOT / "revision" / "results" / "molina_nested_cv"
    output_dir = ROOT / "revision" / "results" / "molina_secondary_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    references = load_references()
    frames = pd.read_csv(historical_frame_table())
    probabilities = pd.read_csv(run_dir / "outer_test_frame_probabilities.csv.gz")
    predictions = pd.read_csv(run_dir / "outer_test_note_predictions.csv")
    per_recording = pd.read_csv(run_dir / "per_recording_metrics.csv")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    frame_micro, frame_per_recording = frame_level_metrics(frames, references)
    frame_per_recording.to_csv(output_dir / "frame_level_per_recording.csv", index=False)
    robustness = parameter_robustness(
        frames, probabilities, references, summary["fold_selections"]
    )
    robustness.to_csv(output_dir / "decoder_parameter_robustness.csv", index=False)
    bootstrap = bootstrap_macro(per_recording)
    bootstrap.to_csv(output_dir / "recording_bootstrap_ci.csv", index=False)
    burden = error_burden(per_recording)
    burden.to_csv(output_dir / "error_burden_per_recording.csv", index=False)

    durations = burden["audio_duration_seconds"].sum()
    aggregate_error_burden = {
        "total_audio_seconds": float(durations),
        "split_per_5_seconds": float(
            burden["split_approx_count"].sum() / durations * 5.0
        ),
        "merged_per_5_seconds": float(
            burden["merged_approx_count"].sum() / durations * 5.0
        ),
        "spurious_per_5_seconds": float(
            burden["spurious_approx_count"].sum() / durations * 5.0
        ),
    }
    output_summary = {
        "formal_note_level": summary["micro"],
        "formal_note_level_macro": summary["macro"],
        "crepe_frontend_frame_level": frame_micro,
        "aggregate_error_burden": aggregate_error_burden,
        "note": (
            "Frame-level reference is rasterized from note GT. Estimated voicing uses "
            "CREPE confidence threshold 0.30, selected in all outer folds."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(output_summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output_summary, indent=2))
    print("\nRobustness:\n", robustness[["setting", "micro_COnPOff_F", "micro_est_notes"]])
    print("\nBootstrap:\n", bootstrap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
