#!/usr/bin/env python3
"""Evaluate the frozen Molina-trained pipeline zero-shot on HumTrans audio."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revision" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jasm_revision.decoder import decode_notes  # noqa: E402
from jasm_revision.evaluation import evaluate_corpus  # noqa: E402
from jasm_revision.features import extract_features_from_wav  # noqa: E402
from jasm_revision.humtrans import midi_file_to_notes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "revision" / "external" / "archives" / "HumTrans-all_wav.zip",
    )
    parser.add_argument(
        "--wav-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--midi-dir",
        type=Path,
        default=ROOT / "revision" / "external" / "extracted" / "humtrans_midi",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-recordings", type=int, default=0)
    return parser.parse_args()


def extract_test_wavs(archive: Path, destination: Path, keys: list[str]) -> dict[str, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    output = {}
    required = {f"{key}.wav" for key in keys}
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            name = Path(member.filename).name
            if name not in required or member.is_dir():
                continue
            target = destination / name
            if not target.exists():
                with handle.open(member) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink, length=1024 * 1024)
            output[name.removesuffix(".wav")] = target
    missing = sorted(set(keys) - set(output))
    if missing:
        raise FileNotFoundError(f"missing HumTrans test wavs: {missing[:10]}")
    return output


def feature_columns(config: dict) -> list[str]:
    columns = list(config["features"]["columns"])
    columns.extend(
        f"{source}_lag{int(lag):+d}"
        for source in config["features"]["lag_sources"]
        for lag in config["features"]["lags"]
    )
    return columns


def main() -> int:
    args = parse_args()
    if args.wav_dir is None:
        args.wav_dir = (
            ROOT
            / "revision"
            / "external"
            / "extracted"
            / f"humtrans_wav_{args.split}"
        )
    if args.output_dir is None:
        suffix = "humtrans_zero_shot" if args.split == "test" else "humtrans_valid_zero_shot"
        args.output_dir = ROOT / "revision" / "results" / suffix
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(
        (ROOT / "revision" / "config" / "main_experiment.yaml").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (
            ROOT
            / "revision"
            / "results"
            / "external_deployment_model"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    output_model_dir = ROOT / "revision" / "results" / "external_deployment_model"
    onset_model = joblib.load(output_model_dir / "onset_model.joblib")
    offset_model = joblib.load(output_model_dir / "offset_model.joblib")
    decoder_parameters = manifest["decoder_parameters"]
    columns = feature_columns(config)

    keys = [
        line.strip()
        for line in (
            ROOT
            / "revision"
            / "external"
            / "repos"
            / "HumTrans-main"
            / f"{args.split}_keys.txt"
        )
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if args.max_recordings:
        keys = keys[: args.max_recordings]
    wav_paths = extract_test_wavs(args.archive, args.wav_dir, keys)
    midi_base = args.midi_dir
    references = {
        key: midi_file_to_notes(
            midi_base / "GroundTruth" / args.split / f"{key}.mid"
        )
        for key in keys
    }
    predictions = {}
    rows = []
    for index, key in enumerate(keys, start=1):
        frames = extract_features_from_wav(
            wav_paths[key],
            key,
            lag_sources=config["features"]["lag_sources"],
            lags=config["features"]["lags"],
            device="cuda" if __import__("torch").cuda.is_available() else "cpu",
            batch_size=32,
        )
        x = (
            frames[columns]
            .replace([np.inf, -np.inf], 0.0)
            .fillna(0.0)
            .to_numpy(np.float32)
        )
        frames["onset_probability"] = onset_model.predict_proba(x)[:, 1]
        frames["offset_probability"] = offset_model.predict_proba(x)[:, 1]
        prediction = decode_notes(frames, **decoder_parameters)
        predictions[key] = prediction
        prediction.to_csv(args.output_dir / f"{key}.notes.csv", index=False)
        rows.append(
            {
                "key": key,
                "frames": len(frames),
                "ref_notes": len(references[key]),
                "est_notes": len(prediction),
            }
        )
        if index % 25 == 0 or index == len(keys):
            print(
                f"HumTrans {args.split} zero-shot: {index}/{len(keys)}", flush=True
            )

    micro, per_recording, macro = evaluate_corpus(references, predictions, keys)
    per_recording.to_csv(args.output_dir / "per_recording_metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(args.output_dir / "inference_audit.csv", index=False)
    summary = {
        "evaluation": f"HumTrans {args.split} zero-shot",
        "split": args.split,
        "recordings": len(keys),
        "training_corpus": "38 Molina recordings only",
        "external_test_tuning": "none",
        "micro_standard_molina_metrics": micro,
        "macro_standard_molina_metrics": macro,
        "decoder_parameters": decoder_parameters,
        "feature_extractor": manifest["feature_extractor"],
        "note": (
            "This is a zero-shot cross-corpus test. HumTrans official baseline "
            "numbers use a separate onset-only octave-invariant metric and are not "
            "combined with this standard Molina table."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
