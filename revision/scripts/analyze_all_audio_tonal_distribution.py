#!/usr/bin/env python3
"""Recompute descriptive tonal labels for all 132 Molina waveforms.

The analysis uses the same full torchcrepe front end and confidence threshold as
the revised manuscript.  Template labels are descriptive outputs only; they are
not evaluated as key predictions because the corpus has no independent key
annotations.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import median_filter


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revision" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jasm_revision.features import (  # noqa: E402
    CREPE_MEDIAN_FILTER,
    CREPE_MODEL,
    FRAME_HOP_SECONDS,
    PITCH_MEDIAN_FILTER,
    extract_crepe_track,
    read_wav_mono,
)
from jasm_revision.tonal import audio_profile, template_label  # noqa: E402


CONFIDENCE_THRESHOLD = 0.30
SEED = 2026


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore per-recording CREPE caches and recompute every waveform.",
    )
    return parser.parse_args()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def load_or_extract(
    wav_path: Path,
    cache_path: Path,
    *,
    device: str,
    batch_size: int,
    overwrite: bool,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    waveform, sample_rate = read_wav_mono(wav_path)
    duration_seconds = len(waveform) / sample_rate
    if cache_path.exists() and not overwrite:
        with np.load(cache_path) as cached:
            midi = cached["midi"].astype(np.float64)
            confidence = cached["confidence"].astype(np.float64)
        return midi, confidence, duration_seconds, "cache"

    track = extract_crepe_track(
        waveform,
        sample_rate,
        device=device,
        batch_size=batch_size,
    )
    midi = median_filter(
        track["midi"].to_numpy(np.float64),
        size=PITCH_MEDIAN_FILTER,
    )
    confidence = track["confidence"].to_numpy(np.float64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        midi=midi.astype(np.float32),
        confidence=confidence.astype(np.float32),
        frame_hop_seconds=np.asarray(FRAME_HOP_SECONDS),
        sample_rate=np.asarray(sample_rate),
    )
    return midi, confidence, duration_seconds, "computed"


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not visible to PyTorch")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    output_dir = ROOT / "revision" / "results" / "all_audio_tonal_distribution"
    cache_dir = output_dir / "crepe_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_paths = sorted((ROOT / "audio").glob("*.wav"))
    if len(wav_paths) != 132:
        raise AssertionError(f"expected 132 waveforms, found {len(wav_paths)}")

    rows: list[dict[str, object]] = []
    for index, wav_path in enumerate(wav_paths, start=1):
        cache_path = cache_dir / f"{wav_path.stem}.npz"
        midi, confidence, duration_seconds, cache_status = load_or_extract(
            wav_path,
            cache_path,
            device=args.device,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
        frames = pd.DataFrame(
            {
                "midi": midi,
                "conf": confidence,
                "voiced": np.ones(len(midi), dtype=np.float64),
            }
        )
        profile = audio_profile(frames, CONFIDENCE_THRESHOLD)
        if profile.sum() <= 0:
            label = {"tonic": -1, "mode": "unknown", "label": "unknown", "score": np.nan}
        else:
            label = template_label(profile)
        accepted = confidence >= CONFIDENCE_THRESHOLD
        rows.append(
            {
                "stem": wav_path.stem,
                "duration_seconds": duration_seconds,
                "frames": len(midi),
                "accepted_frames": int(accepted.sum()),
                "accepted_frame_fraction": float(accepted.mean()),
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "template_label": label["label"],
                "tonic": int(label["tonic"]),
                "mode": label["mode"],
                "template_score": float(label["score"]),
                "cache_status": cache_status,
                **{f"pc_{pc}": float(profile[pc]) for pc in range(12)},
            }
        )
        print(
            f"[{index:03d}/{len(wav_paths)}] {wav_path.stem}: "
            f"{label['label']} ({cache_status})",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    per_recording = pd.DataFrame(rows).sort_values("stem").reset_index(drop=True)
    per_recording.to_csv(output_dir / "per_recording_tonal_labels.csv", index=False)
    mode_summary = (
        per_recording.groupby("mode", dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    mode_summary["percent"] = mode_summary["count"] / len(per_recording) * 100.0
    mode_summary.to_csv(output_dir / "mode_summary.csv", index=False)
    tonic_mode = (
        per_recording.groupby(["tonic", "mode"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    tonic_mode.to_csv(output_dir / "tonic_mode_summary.csv", index=False)

    summary = {
        "analysis_type": "descriptive_template_labels_not_key_accuracy",
        "recordings": int(len(per_recording)),
        "total_audio_seconds": float(per_recording["duration_seconds"].sum()),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "pitch_frontend": {
            "implementation": "torchcrepe",
            "model": CREPE_MODEL,
            "frame_hop_seconds": FRAME_HOP_SECONDS,
            "crepe_median_filter_frames": CREPE_MEDIAN_FILTER,
            "midi_median_filter_frames": PITCH_MEDIAN_FILTER,
        },
        "mode_counts": {
            str(row.mode): int(row.count) for row in mode_summary.itertuples(index=False)
        },
        "interpretation_guardrail": (
            "Krumhansl-Schmuckler labels summarize the audio-derived pitch-class "
            "profiles. They are not key-prediction accuracy estimates because no "
            "independent key annotations are available."
        ),
        "seed": SEED,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "torchcrepe": package_version("torchcrepe"),
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
