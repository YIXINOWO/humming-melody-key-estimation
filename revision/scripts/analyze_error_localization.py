#!/usr/bin/env python3
"""Test whether Molina segmentation errors concentrate near phrase boundaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revision" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jasm_revision.error_localization import (  # noqa: E402
    boundary_window_coverage,
    distance_to_boundaries,
    phrase_boundaries,
    segmentation_error_events,
)
from jasm_revision.ground_truth import read_ground_truth  # noqa: E402


SEED = 2026
PERMUTATIONS = 10000
PRIMARY_GAP_SECONDS = 0.30
PRIMARY_WINDOW_SECONDS = 0.15


def load_data() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    references = {
        path.name.removesuffix(".GroundTruth.txt"): read_ground_truth(path).notes
        for path in sorted((ROOT / "gt_files_temp").glob("*.GroundTruth.txt"))
    }
    predictions = pd.read_csv(
        ROOT
        / "revision"
        / "results"
        / "molina_nested_cv"
        / "outer_test_note_predictions.csv"
    )
    estimates = {
        stem: part.drop(columns="stem").reset_index(drop=True)
        for stem, part in predictions.groupby("stem", sort=True)
    }
    return references, estimates


def analyze_setting(
    references: dict[str, pd.DataFrame],
    estimates: dict[str, pd.DataFrame],
    gap_seconds: float,
    window_seconds: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_parts = []
    recording_rows = []
    boundaries_by_stem = {}
    spans = {}
    for stem in sorted(references):
        reference = references[stem]
        estimate = estimates[stem]
        boundaries = phrase_boundaries(reference, gap_seconds)
        span_start = float(reference["onset"].min())
        span_end = float(reference["offset"].max())
        events = segmentation_error_events(stem, reference, estimate)
        events["distance_to_phrase_boundary_seconds"] = distance_to_boundaries(
            events["event_time"].to_numpy(np.float64), boundaries
        )
        events["near_phrase_boundary"] = (
            events["distance_to_phrase_boundary_seconds"] <= window_seconds
        )
        event_parts.append(events)
        boundaries_by_stem[stem] = boundaries
        spans[stem] = (span_start, span_end)
        recording_rows.append(
            {
                "stem": stem,
                "phrase_boundaries": len(boundaries),
                "reference_span_seconds": span_end - span_start,
                "boundary_window_coverage": boundary_window_coverage(
                    span_start, span_end, boundaries, window_seconds
                ),
                **{
                    f"{error_type}_events": int(
                        (events["error_type"] == error_type).sum()
                    )
                    for error_type in ("split", "merged", "spurious")
                },
            }
        )

    all_events = pd.concat(event_parts, ignore_index=True)
    rows = []
    for error_type in ("split", "merged", "spurious", "all"):
        selected = (
            all_events
            if error_type == "all"
            else all_events.loc[all_events["error_type"] == error_type]
        )
        observed = int(selected["near_phrase_boundary"].sum())
        total = len(selected)
        permuted_counts = np.zeros(PERMUTATIONS, dtype=np.int32)
        expected_count = 0.0
        for stem, part in selected.groupby("stem", sort=True):
            times = part["event_time"].to_numpy(np.float64)
            start, end = spans[stem]
            duration = end - start
            expected_count += len(times) * boundary_window_coverage(
                start, end, boundaries_by_stem[stem], window_seconds
            )
            if duration <= 0 or not len(times):
                continue
            offsets = rng.uniform(0.0, duration, size=PERMUTATIONS)
            shifted = (
                times[None, :] - start + offsets[:, None]
            ) % duration + start
            distances = np.min(
                np.abs(
                    shifted[:, :, None]
                    - boundaries_by_stem[stem][None, None, :]
                ),
                axis=2,
            )
            permuted_counts += np.sum(distances <= window_seconds, axis=1)
        p_value = float((1 + np.sum(permuted_counts >= observed)) / (PERMUTATIONS + 1))
        rows.append(
            {
                "gap_threshold_seconds": gap_seconds,
                "boundary_window_seconds": window_seconds,
                "error_type": error_type,
                "events": total,
                "near_boundary_events": observed,
                "observed_near_fraction": observed / total if total else np.nan,
                "time_coverage_expected_count": expected_count,
                "time_coverage_expected_fraction": expected_count / total
                if total
                else np.nan,
                "enrichment_ratio": observed / expected_count
                if expected_count > 0
                else np.nan,
                "circular_shift_p_value_one_sided": p_value,
                "permutations": PERMUTATIONS,
            }
        )
    return all_events, pd.DataFrame(rows).merge(
        pd.DataFrame(recording_rows)[
            ["stem", "phrase_boundaries", "boundary_window_coverage"]
        ].agg(
            {
                "phrase_boundaries": "sum",
                "boundary_window_coverage": "mean",
            }
        ).to_frame().T.assign(join_key=1),
        how="cross",
    ).drop(columns="join_key", errors="ignore")


def main() -> int:
    output_dir = ROOT / "revision" / "results" / "molina_error_localization"
    output_dir.mkdir(parents=True, exist_ok=True)
    references, estimates = load_data()
    rng = np.random.default_rng(SEED)
    sensitivity_parts = []
    primary_events = None
    for gap_seconds in (0.20, 0.30, 0.50):
        for window_seconds in (0.10, 0.15, 0.20):
            events, summary = analyze_setting(
                references,
                estimates,
                gap_seconds,
                window_seconds,
                rng,
            )
            sensitivity_parts.append(summary)
            if (
                gap_seconds == PRIMARY_GAP_SECONDS
                and window_seconds == PRIMARY_WINDOW_SECONDS
            ):
                primary_events = events
    sensitivity = pd.concat(sensitivity_parts, ignore_index=True)
    primary = sensitivity.loc[
        (sensitivity["gap_threshold_seconds"] == PRIMARY_GAP_SECONDS)
        & (sensitivity["boundary_window_seconds"] == PRIMARY_WINDOW_SECONDS)
    ].reset_index(drop=True)
    assert primary_events is not None
    primary_events.to_csv(output_dir / "error_events.csv", index=False)
    sensitivity.to_csv(output_dir / "boundary_sensitivity.csv", index=False)

    summary = {
        "operational_definition": {
            "phrase_boundary": (
                "reference-sequence start/end and both sides of an inter-note gap "
                f">= {PRIMARY_GAP_SECONDS:.2f} s"
            ),
            "near_boundary": (
                f"event time within {PRIMARY_WINDOW_SECONDS:.2f} s of a phrase boundary"
            ),
            "split_event": "onset of each excess prediction overlapping one reference note",
            "merged_event": "each swallowed reference onset inside one estimated note",
            "spurious_event": "onset of an estimate with no temporal reference overlap",
        },
        "primary_results": primary.to_dict(orient="records"),
        "null_model": (
            "10,000 within-recording circular shifts of each error event pattern; "
            "one-sided p tests enrichment near fixed phrase boundaries"
        ),
        "sensitivity_grid": {
            "gap_threshold_seconds": [0.20, 0.30, 0.50],
            "boundary_window_seconds": [0.10, 0.15, 0.20],
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Molina segmentation-error localization",
        "",
        summary["operational_definition"]["phrase_boundary"] + ".",
        summary["operational_definition"]["near_boundary"] + ".",
        "",
        "| Error | Events | Near boundary | Observed fraction | Time-coverage null | Enrichment | Circular-shift p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary.to_dict(orient="records"):
        lines.append(
            "| {error_type} | {events} | {near_boundary_events} | "
            "{observed_near_fraction:.3f} | {time_coverage_expected_fraction:.3f} | "
            "{enrichment_ratio:.2f} | {circular_shift_p_value_one_sided:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "The boundary definition and window were varied over the full sensitivity "
            "grid in `boundary_sensitivity.csv`; the primary setting was fixed before "
            "inspecting the results.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(primary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
