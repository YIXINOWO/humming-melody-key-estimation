#!/usr/bin/env python3
"""Audit source GT files without modifying source data or historical results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revision" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jasm_revision.ground_truth import read_ground_truth  # noqa: E402


def main() -> int:
    config_path = ROOT / "revision" / "config" / "main_experiment.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    gt_dir = ROOT / config["data"]["ground_truth_dir"]
    audio_dir = ROOT / config["data"]["audio_dir"]
    output_dir = ROOT / config["outputs"]["data_audit"]
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for gt_path in sorted(gt_dir.glob("*.GroundTruth.txt")):
        stem = gt_path.name.removesuffix(".GroundTruth.txt")
        audio_path = audio_dir / f"{stem}.wav"
        record = read_ground_truth(gt_path)
        audio_info = sf.info(audio_path) if audio_path.exists() else None
        audio_duration = float(audio_info.duration) if audio_info else None
        max_offset = float(record.notes["offset"].max())
        rows.append(
            {
                "stem": stem,
                "layout": record.layout,
                "notes": len(record.notes),
                "nonempty_lines": record.nonempty_lines,
                "numeric_tokens": record.numeric_tokens,
                "first_onset_seconds": float(record.notes["onset"].min()),
                "last_offset_seconds": max_offset,
                "audio_exists": audio_path.exists(),
                "audio_duration_seconds": audio_duration,
                "annotation_within_audio": bool(
                    audio_duration is not None and max_offset <= audio_duration + 0.05
                ),
                "adjacent_overlap_count": record.overlap_count,
                "sha256": record.sha256,
            }
        )

    audit = pd.DataFrame(rows).sort_values("stem").reset_index(drop=True)
    expected_recordings = int(config["data"]["expected_recordings"])
    summary = {
        "recordings": int(len(audit)),
        "expected_recordings": expected_recordings,
        "total_reference_notes": int(audit["notes"].sum()),
        "note_row_files": int((audit["layout"] == "note_rows").sum()),
        "value_vector_files": int((audit["layout"] == "value_vector").sum()),
        "missing_audio_files": int((~audit["audio_exists"]).sum()),
        "annotations_outside_audio": int((~audit["annotation_within_audio"]).sum()),
        "files_with_adjacent_overlaps": int((audit["adjacent_overlap_count"] > 0).sum()),
        "audit_passed": bool(
            len(audit) == expected_recordings
            and audit["audio_exists"].all()
            and audit["annotation_within_audio"].all()
        ),
    }

    audit.to_csv(output_dir / "ground_truth_audit.csv", index=False)
    (output_dir / "ground_truth_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    vector_rows = audit.loc[
        audit["layout"] == "value_vector", ["stem", "notes", "nonempty_lines"]
    ]
    report = [
        "# Ground-truth data audit",
        "",
        "This report is generated from source annotations by the tested,",
        "layout-aware parser. No source file is modified.",
        "",
        "## Summary",
        "",
        *(f"- {key}: `{value}`" for key, value in summary.items()),
        "",
        "## Non-standard value-vector files",
        "",
        vector_rows.to_markdown(index=False),
        "",
        "## Full audit table",
        "",
        audit.drop(columns=["sha256"]).to_markdown(index=False),
        "",
    ]
    (output_dir / "ground_truth_audit.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote audit artifacts to {output_dir}")
    return 0 if summary["audit_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

