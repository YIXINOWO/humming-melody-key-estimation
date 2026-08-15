#!/usr/bin/env python3
"""Reproduce HumTrans baselines and rescore them with standard metrics."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revision" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jasm_revision.evaluation import evaluate_corpus  # noqa: E402
from jasm_revision.humtrans import (  # noqa: E402
    evaluate_humtrans_recording,
    midi_file_to_notes,
    trim_estimates_to_reference_span,
)


MODELS = ["VOCANO", "SheetSage", "MIR-ST500", "JDC-STP"]
SPLITS = ["valid", "test"]
OFFICIAL_README_PERCENT = {
    ("VOCANO", "valid"): (3.270, 3.314, 3.194),
    ("VOCANO", "test"): (3.384, 3.329, 3.352),
    ("SheetSage", "valid"): (2.757, 2.656, 2.702),
    ("SheetSage", "test"): (3.039, 2.982, 3.005),
    ("MIR-ST500", "valid"): (6.258, 6.448, 6.341),
    ("MIR-ST500", "test"): (5.686, 5.853, 5.755),
    ("JDC-STP", "valid"): (6.777, 6.785, 6.741),
    ("JDC-STP", "test"): (5.844, 5.620, 5.667),
}
KNOWN_README_DISCREPANCIES = {
    ("VOCANO", "valid", "recall"): (
        "The official repository script returns 3.134%, while the README table "
        "prints 3.314%; precision and F1 reproduce exactly."
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=ROOT / "revision" / "external" / "repos" / "HumTrans-main",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=ROOT / "revision" / "external" / "extracted" / "humtrans_midi",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "revision" / "results" / "humtrans_official_baselines",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            if member.is_dir() or member.filename.startswith("__MACOSX/"):
                continue
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"unsafe archive member: {member.filename}")
            handle.extract(member, destination)


def extract_archives(repo: Path, destination: Path) -> None:
    required = ["GroundTruth", *MODELS]
    for name in required:
        expected = destination / name
        if not expected.exists():
            safe_extract(repo / "midis" / f"{name}.zip", destination)


def split_keys(repo: Path, split: str) -> list[str]:
    return [
        line.strip()
        for line in (repo / f"{split}_keys.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_files(base: Path, model: str, split: str, keys: list[str]) -> None:
    missing_reference = [
        key for key in keys if not (base / "GroundTruth" / split / f"{key}.mid").is_file()
    ]
    missing_estimate = [
        key for key in keys if not (base / model / split / f"{key}.mid").is_file()
    ]
    if missing_reference or missing_estimate:
        raise FileNotFoundError(
            f"{model}/{split}: missing {len(missing_reference)} references and "
            f"{len(missing_estimate)} estimates"
        )


def evaluate_model_split(
    base: Path, model: str, split: str, keys: list[str]
) -> tuple[pd.DataFrame, dict]:
    reference_by_key = {}
    estimate_by_key = {}
    rows = []
    for index, key in enumerate(keys, start=1):
        reference = midi_file_to_notes(base / "GroundTruth" / split / f"{key}.mid")
        estimate = midi_file_to_notes(base / model / split / f"{key}.mid")
        metrics = evaluate_humtrans_recording(reference, estimate)
        rows.append({"model": model, "split": split, "key": key, **metrics})
        reference_by_key[key] = reference
        estimate_by_key[key] = trim_estimates_to_reference_span(reference, estimate)
        if index % 200 == 0:
            print(f"{model}/{split}: {index}/{len(keys)}", flush=True)

    per_recording = pd.DataFrame(rows)
    micro, _, macro = evaluate_corpus(reference_by_key, estimate_by_key, keys)
    official = per_recording[
        ["official_precision", "official_recall", "official_f1"]
    ].mean()
    reproduced_percent = tuple(float(value * 100.0) for value in official)
    published_percent = OFFICIAL_README_PERCENT[(model, split)]
    metric_names = ["precision", "recall", "f1"]
    absolute_differences = {
        metric: abs(reproduced - published)
        for metric, reproduced, published in zip(
            metric_names, reproduced_percent, published_percent, strict=True
        )
    }
    known_discrepancies = {
        metric: KNOWN_README_DISCREPANCIES[(model, split, metric)]
        for metric in metric_names
        if (model, split, metric) in KNOWN_README_DISCREPANCIES
    }
    unexplained_differences = {
        metric: difference
        for metric, difference in absolute_differences.items()
        if difference > 0.001 and metric not in known_discrepancies
    }
    summary = {
        "model": model,
        "split": split,
        "recordings": len(keys),
        "official_macro_fraction": {
            "precision": float(official["official_precision"]),
            "recall": float(official["official_recall"]),
            "f1": float(official["official_f1"]),
        },
        "official_macro_percent_reproduced": dict(
            zip(["precision", "recall", "f1"], reproduced_percent, strict=True)
        ),
        "official_readme_percent": dict(
            zip(["precision", "recall", "f1"], published_percent, strict=True)
        ),
        "readme_abs_difference_percent": absolute_differences,
        "known_readme_discrepancies": known_discrepancies,
        "official_reproduction_pass": not unexplained_differences,
        "standard_micro": micro,
        "standard_macro": macro,
    }
    return per_recording, summary


def write_report(output_dir: Path, summaries: list[dict]) -> None:
    lines = [
        "# HumTrans official-baseline audit",
        "",
        "The official HumTrans onset-only, octave-invariant metric is reproduced "
        "separately from the standard Molina-style note metrics.",
        "",
        "| Model | Split | N | Official F1 (%) reproduced | README F1 (%) | Standard COnPOff micro | Standard COnPOff macro | Reproduction gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        lines.append(
            "| {model} | {split} | {recordings} | {official:.3f} | {published:.3f} | "
            "{micro:.4f} | {macro:.4f} | {gate} |".format(
                model=item["model"],
                split=item["split"],
                recordings=item["recordings"],
                official=item["official_macro_percent_reproduced"]["f1"],
                published=item["official_readme_percent"]["f1"],
                micro=item["standard_micro"]["COnPOff_F"],
                macro=item["standard_macro"]["COnPOff_F"],
                gate="PASS" if item["official_reproduction_pass"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "The official source script independently returns 3.134% recall for "
            "VOCANO/valid, whereas the README table prints 3.314%. Its precision "
            "and F1 reproduce exactly, so this is retained as a documented source "
            "table discrepancy rather than silently corrected.",
            "",
            "Do not place the official HumTrans percentages in the same metric column "
            "as Molina COnPOff. The former ignores offsets and searches over global "
            "octave shifts with a 1-cent pitch tolerance.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extract_archives(args.repo, args.extract_dir)

    all_rows = []
    summaries = []
    for model in MODELS:
        for split in SPLITS:
            keys = split_keys(args.repo, split)
            validate_files(args.extract_dir, model, split, keys)
            per_recording, summary = evaluate_model_split(
                args.extract_dir, model, split, keys
            )
            all_rows.append(per_recording)
            summaries.append(summary)
            print(json.dumps(summary, indent=2), flush=True)

    pd.concat(all_rows, ignore_index=True).to_csv(
        args.output_dir / "per_recording_metrics.csv.gz", index=False
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps({"evaluations": summaries}, indent=2) + "\n", encoding="utf-8"
    )
    write_report(args.output_dir, summaries)
    if not all(item["official_reproduction_pass"] for item in summaries):
        raise RuntimeError("one or more official HumTrans reproduction gates failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
