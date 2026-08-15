"""Leakage-free grouped nested cross-validation for the revised main result."""

from __future__ import annotations

import itertools
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mir_eval
import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from .decoder import decode_notes
from .evaluation import conpoff_match_counts, evaluate_corpus, f_measure_from_counts
from .ground_truth import read_ground_truth
from .labels import make_boundary_labels


@dataclass(frozen=True)
class FoldSelection:
    outer_fold: int
    model_candidate: int
    model_parameters: dict[str, Any]
    decoder_parameters: dict[str, float]
    inner_model_score: float
    inner_decoder_conpoff: float
    train_stems: list[str]
    test_stems: list[str]


def grouped_splits(
    groups: np.ndarray, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    placeholder = np.zeros(len(groups), dtype=np.int8)
    return list(splitter.split(placeholder, groups=groups))


def assert_group_isolation(
    groups: np.ndarray, train_indices: np.ndarray, test_indices: np.ndarray
) -> None:
    train_groups = set(groups[train_indices])
    test_groups = set(groups[test_indices])
    overlap = train_groups & test_groups
    if overlap:
        raise AssertionError(f"group leakage: {sorted(overlap)}")


def feature_columns(config: dict[str, Any], available: list[str]) -> list[str]:
    columns = list(config["features"]["columns"])
    for source in config["features"]["lag_sources"]:
        for lag in config["features"]["lags"]:
            columns.append(f"{source}_lag{int(lag):+d}")
    missing = sorted(set(columns) - set(available))
    if missing:
        raise KeyError(f"historical feature table is missing columns: {missing}")
    return columns


def _fit_probability_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    parameters: dict[str, Any],
    seed: int,
) -> np.ndarray:
    model = RandomForestClassifier(**parameters, random_state=seed)
    model.fit(x_train, y_train)
    classes = model.classes_.tolist()
    if 1 not in classes:
        return np.zeros(len(x_test), dtype=np.float64)
    positive_index = classes.index(1)
    return model.predict_proba(x_test)[:, positive_index]


def _candidate_parameters(config: dict[str, Any], smoke_test: bool) -> list[dict[str, Any]]:
    common = dict(config["random_forest"]["common"])
    candidates = []
    for candidate in config["random_forest"]["candidates"]:
        merged = {**common, **candidate}
        if smoke_test:
            merged.update(n_estimators=20, max_depth=8, min_samples_leaf=8, n_jobs=8)
            return [merged]
        candidates.append(merged)
    return candidates


def _decoder_candidates(config: dict[str, Any], smoke_test: bool) -> list[dict[str, float]]:
    grid = config["decoder_grid"]
    keys = list(grid)
    values = [grid[key] for key in keys]
    candidates = [
        {key: float(value) for key, value in zip(keys, combination, strict=True)}
        for combination in itertools.product(*values)
    ]
    if smoke_test:
        baseline = {
            "onset_threshold": 0.40,
            "offset_threshold": 0.45,
            "min_onset_separation_seconds": 0.18,
            "min_note_duration_seconds": 0.10,
            "crepe_confidence_threshold": 0.25,
        }
        sensitivity = {**baseline, "onset_threshold": 0.44}
        return [baseline, sensitivity]
    return candidates


def _decode_stems(
    frame_table: pd.DataFrame,
    probabilities: pd.DataFrame,
    stems: list[str],
    decoder_parameters: dict[str, float],
) -> dict[str, pd.DataFrame]:
    probability_lookup = probabilities.set_index("row_index")
    predictions: dict[str, pd.DataFrame] = {}
    for stem in stems:
        part = frame_table.loc[frame_table["stem"] == stem].copy()
        part["onset_probability"] = probability_lookup.loc[
            part.index, "onset_probability"
        ].to_numpy(dtype=np.float64)
        part["offset_probability"] = probability_lookup.loc[
            part.index, "offset_probability"
        ].to_numpy(dtype=np.float64)
        predictions[stem] = decode_notes(part, **decoder_parameters)
    return predictions


def _select_decoder(
    frame_table: pd.DataFrame,
    probabilities: pd.DataFrame,
    reference_by_stem: dict[str, pd.DataFrame],
    stems: list[str],
    candidates: list[dict[str, float]],
) -> tuple[dict[str, float], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        estimates = _decode_stems(frame_table, probabilities, stems, candidate)
        matches = references = estimated = 0
        for stem in stems:
            local_matches, local_references, local_estimated = conpoff_match_counts(
                reference_by_stem[stem], estimates[stem]
            )
            matches += local_matches
            references += local_references
            estimated += local_estimated
        score = f_measure_from_counts(matches, references, estimated)
        rows.append(
            {
                "decoder_candidate": candidate_index,
                **candidate,
                "matches": matches,
                "ref_notes": references,
                "est_notes": estimated,
                "COnPOff_F_micro": score,
            }
        )
    search = pd.DataFrame(rows).sort_values(
        ["COnPOff_F_micro", "est_notes", "decoder_candidate"],
        ascending=[False, True, True],
    )
    best_index = int(search.iloc[0]["decoder_candidate"])
    return candidates[best_index], search.reset_index(drop=True)


def _versions() -> dict[str, str]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "mir_eval": mir_eval.__version__,
    }


def run_nested_cv(
    root: Path,
    config_path: Path,
    *,
    smoke_test: bool = False,
    overwrite: bool = False,
) -> Path:
    started = time.time()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["project"]["seed"])
    outer_splits = 2 if smoke_test else int(config["evaluation"]["outer_splits"])
    inner_splits = 2 if smoke_test else int(config["evaluation"]["inner_splits"])
    output_dir = root / "revision" / "results" / (
        "smoke_nested_cv" if smoke_test else "molina_nested_cv"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = root / config["data"]["ground_truth_dir"]
    reference_by_stem = {
        path.name.removesuffix(".GroundTruth.txt"): read_ground_truth(path).notes
        for path in sorted(gt_dir.glob("*.GroundTruth.txt"))
    }
    frame_path = root / config["data"]["historical_frame_table"]
    frame_table = pd.read_csv(frame_path)
    frame_table.index.name = "row_index"
    frame_stems = set(frame_table["stem"].unique())
    if frame_stems != set(reference_by_stem):
        raise AssertionError(
            f"frame/GT stem mismatch: only frames={sorted(frame_stems-set(reference_by_stem))}, "
            f"only GT={sorted(set(reference_by_stem)-frame_stems)}"
        )

    columns = feature_columns(config, frame_table.columns.tolist())
    x = (
        frame_table[columns]
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    corrected = make_boundary_labels(frame_table, reference_by_stem, radius_frames=1)
    y_onset = corrected["onset_label"].to_numpy(dtype=np.int8)
    y_offset = corrected["offset_label"].to_numpy(dtype=np.int8)
    groups = frame_table["stem"].to_numpy()

    label_audit = (
        pd.DataFrame(
            {
                "stem": groups,
                "old_onset_label": frame_table["onset_label"].to_numpy(dtype=np.int8),
                "new_onset_label": y_onset,
                "old_offset_label": frame_table["offset_label"].to_numpy(dtype=np.int8),
                "new_offset_label": y_offset,
            }
        )
        .groupby("stem")
        .agg(
            old_onset_positives=("old_onset_label", "sum"),
            new_onset_positives=("new_onset_label", "sum"),
            old_offset_positives=("old_offset_label", "sum"),
            new_offset_positives=("new_offset_label", "sum"),
        )
        .reset_index()
    )
    label_audit.to_csv(output_dir / "corrected_label_audit.csv", index=False)

    model_candidates = _candidate_parameters(config, smoke_test)
    decoder_candidates = _decoder_candidates(config, smoke_test)
    outer = grouped_splits(groups, outer_splits, seed)
    assignments: list[dict[str, Any]] = []
    all_test_probabilities: list[pd.DataFrame] = []
    all_test_predictions: list[pd.DataFrame] = []
    selections: list[FoldSelection] = []

    for outer_number, (outer_train, outer_test) in enumerate(outer, start=1):
        assert_group_isolation(groups, outer_train, outer_test)
        fold_dir = output_dir / f"fold_{outer_number}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        complete_flag = fold_dir / "complete.json"
        if complete_flag.exists() and not overwrite:
            selection_data = json.loads(complete_flag.read_text(encoding="utf-8"))
            selections.append(FoldSelection(**selection_data))
            all_test_probabilities.append(pd.read_csv(fold_dir / "test_frame_probabilities.csv.gz"))
            all_test_predictions.append(pd.read_csv(fold_dir / "test_note_predictions.csv"))
            continue

        train_stems = sorted(set(groups[outer_train]))
        test_stems = sorted(set(groups[outer_test]))
        for stem in train_stems:
            assignments.append({"outer_fold": outer_number, "stem": stem, "role": "train"})
        for stem in test_stems:
            assignments.append({"outer_fold": outer_number, "stem": stem, "role": "test"})
        print(
            f"[outer {outer_number}/{outer_splits}] train={len(train_stems)} "
            f"test={len(test_stems)} frames={len(outer_train)}/{len(outer_test)}",
            flush=True,
        )

        inner_groups = groups[outer_train]
        inner = grouped_splits(inner_groups, inner_splits, seed + outer_number * 100)
        model_rows: list[dict[str, Any]] = []
        candidate_probabilities: list[pd.DataFrame] = []
        for candidate_number, parameters in enumerate(model_candidates):
            onset_probability = np.zeros(len(outer_train), dtype=np.float64)
            offset_probability = np.zeros(len(outer_train), dtype=np.float64)
            for inner_number, (inner_train_local, inner_validation_local) in enumerate(
                inner, start=1
            ):
                assert_group_isolation(
                    inner_groups, inner_train_local, inner_validation_local
                )
                train_indices = outer_train[inner_train_local]
                validation_indices = outer_train[inner_validation_local]
                onset_probability[inner_validation_local] = _fit_probability_model(
                    x[train_indices],
                    y_onset[train_indices],
                    x[validation_indices],
                    parameters,
                    seed + outer_number * 10000 + candidate_number * 100 + inner_number,
                )
                offset_probability[inner_validation_local] = _fit_probability_model(
                    x[train_indices],
                    y_offset[train_indices],
                    x[validation_indices],
                    parameters,
                    seed
                    + 500000
                    + outer_number * 10000
                    + candidate_number * 100
                    + inner_number,
                )
            onset_ap = float(average_precision_score(y_onset[outer_train], onset_probability))
            offset_ap = float(average_precision_score(y_offset[outer_train], offset_probability))
            score = (onset_ap + offset_ap) / 2.0
            model_rows.append(
                {
                    "model_candidate": candidate_number,
                    **parameters,
                    "onset_average_precision": onset_ap,
                    "offset_average_precision": offset_ap,
                    "mean_frame_average_precision": score,
                }
            )
            candidate_probabilities.append(
                pd.DataFrame(
                    {
                        "row_index": outer_train,
                        "onset_probability": onset_probability,
                        "offset_probability": offset_probability,
                    }
                )
            )
            print(
                f"  model {candidate_number}: onset_AP={onset_ap:.4f} "
                f"offset_AP={offset_ap:.4f}",
                flush=True,
            )

        model_search = pd.DataFrame(model_rows).sort_values(
            ["mean_frame_average_precision", "model_candidate"], ascending=[False, True]
        )
        model_search.to_csv(fold_dir / "model_selection.csv", index=False)
        selected_model_number = int(model_search.iloc[0]["model_candidate"])
        selected_parameters = model_candidates[selected_model_number]
        selected_inner_probabilities = candidate_probabilities[selected_model_number]
        selected_decoder, decoder_search = _select_decoder(
            frame_table,
            selected_inner_probabilities,
            reference_by_stem,
            train_stems,
            decoder_candidates,
        )
        decoder_search.to_csv(fold_dir / "decoder_selection.csv", index=False)

        onset_test_probability = _fit_probability_model(
            x[outer_train],
            y_onset[outer_train],
            x[outer_test],
            selected_parameters,
            seed + 900000 + outer_number,
        )
        offset_test_probability = _fit_probability_model(
            x[outer_train],
            y_offset[outer_train],
            x[outer_test],
            selected_parameters,
            seed + 950000 + outer_number,
        )
        test_probabilities = pd.DataFrame(
            {
                "row_index": outer_test,
                "stem": groups[outer_test],
                "onset_label": y_onset[outer_test],
                "offset_label": y_offset[outer_test],
                "onset_probability": onset_test_probability,
                "offset_probability": offset_test_probability,
            }
        ).sort_values("row_index")
        test_probabilities.to_csv(
            fold_dir / "test_frame_probabilities.csv.gz", index=False, compression="gzip"
        )
        estimates = _decode_stems(
            frame_table, test_probabilities, test_stems, selected_decoder
        )
        prediction_parts = []
        for stem in test_stems:
            part = estimates[stem].copy()
            part.insert(0, "stem", stem)
            prediction_parts.append(part)
        test_predictions = pd.concat(prediction_parts, ignore_index=True)
        test_predictions.to_csv(fold_dir / "test_note_predictions.csv", index=False)

        selection = FoldSelection(
            outer_fold=outer_number,
            model_candidate=selected_model_number,
            model_parameters=selected_parameters,
            decoder_parameters=selected_decoder,
            inner_model_score=float(model_search.iloc[0]["mean_frame_average_precision"]),
            inner_decoder_conpoff=float(decoder_search.iloc[0]["COnPOff_F_micro"]),
            train_stems=train_stems,
            test_stems=test_stems,
        )
        complete_flag.write_text(
            json.dumps(asdict(selection), indent=2) + "\n", encoding="utf-8"
        )
        selections.append(selection)
        all_test_probabilities.append(test_probabilities)
        all_test_predictions.append(test_predictions)
        print(
            f"  selected model={selected_model_number} decoder={selected_decoder}",
            flush=True,
        )

    probabilities = pd.concat(all_test_probabilities, ignore_index=True).sort_values(
        "row_index"
    )
    if probabilities["row_index"].duplicated().any() or len(probabilities) != len(frame_table):
        raise AssertionError("outer predictions do not cover each frame exactly once")
    probabilities.to_csv(
        output_dir / "outer_test_frame_probabilities.csv.gz",
        index=False,
        compression="gzip",
    )
    predictions = pd.concat(all_test_predictions, ignore_index=True)
    predictions.to_csv(output_dir / "outer_test_note_predictions.csv", index=False)
    estimate_by_stem = {
        stem: group.drop(columns=["stem"]).reset_index(drop=True)
        for stem, group in predictions.groupby("stem")
    }
    for stem in reference_by_stem:
        estimate_by_stem.setdefault(
            stem, pd.DataFrame(columns=["onset", "offset", "midi", "hz"])
        )
    ordered_stems = sorted(reference_by_stem)
    micro, per_recording, macro = evaluate_corpus(
        reference_by_stem, estimate_by_stem, ordered_stems
    )
    per_recording.to_csv(output_dir / "per_recording_metrics.csv", index=False)
    pd.DataFrame(assignments).to_csv(output_dir / "outer_fold_assignments.csv", index=False)
    summary = {
        "run_type": "smoke_test" if smoke_test else "formal_grouped_nested_cv",
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "reference_notes": int(sum(len(notes) for notes in reference_by_stem.values())),
        "micro": micro,
        "macro": macro,
        "fold_selections": [asdict(selection) for selection in selections],
        "versions": _versions(),
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"micro": micro, "macro": macro}, indent=2), flush=True)
    return output_dir
