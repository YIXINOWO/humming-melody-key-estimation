#!/usr/bin/env python3
"""Prepare and evaluate the official ROSVOT baseline on Molina recordings."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revision" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jasm_revision.evaluation import evaluate_corpus  # noqa: E402
from jasm_revision.ground_truth import read_ground_truth  # noqa: E402
from jasm_revision.humtrans import midi_file_to_notes  # noqa: E402


OUTPUT = ROOT / "revision" / "results" / "rosvot_molina"
INFERENCE = OUTPUT / "inference"
ROSVOT_ROOT = ROOT / "revision" / "external" / "repos" / "ROSVOT-main"
ARCHIVE = ROOT / "revision" / "external" / "checkpoints.zip"
NOTE_BOUNDARY_THRESHOLD = 0.85


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_stems() -> list[str]:
    paths = sorted((ROOT / "gt_files_temp").glob("*.GroundTruth.txt"))
    if len(paths) != 38:
        raise RuntimeError(f"Expected 38 Molina annotations, found {len(paths)}")
    return [path.name.removesuffix(".GroundTruth.txt") for path in paths]


def prepare_manifest(stems: list[str]) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for stem in stems:
        wav = (ROOT / "audio" / f"{stem}.wav").resolve()
        if not wav.is_file():
            raise FileNotFoundError(wav)
        manifest.append({"item_name": stem, "wav_fn": str(wav)})
    path = OUTPUT / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def inference_environment() -> dict[str, object]:
    conda = shutil.which("conda")
    if conda is None:
        raise RuntimeError("conda executable not found on PATH")
    code = (
        "import json, platform, torch, librosa, pyworld, pretty_midi; "
        "print(json.dumps({'python': platform.python_version(), "
        "'torch': torch.__version__, 'torch_cuda': torch.version.cuda, "
        "'cuda_available': torch.cuda.is_available(), "
        "'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "
        "'librosa': librosa.__version__, 'pyworld': pyworld.__version__, "
        "'pretty_midi': pretty_midi.__version__}))"
    )
    completed = subprocess.run(
        [conda, "run", "-n", "rosvot-inference", "python", "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write the 38-recording ROSVOT manifest without evaluating predictions.",
    )
    args = parser.parse_args()

    stems = ordered_stems()
    manifest_path = prepare_manifest(stems)
    if args.prepare_only:
        print(manifest_path)
        return 0

    references = {
        stem: read_ground_truth(ROOT / "gt_files_temp" / f"{stem}.GroundTruth.txt").notes
        for stem in stems
    }
    estimates: dict[str, pd.DataFrame] = {}
    prediction_parts = []
    missing = []
    for stem in stems:
        midi_path = INFERENCE / "midi" / f"{stem}.mid"
        if not midi_path.is_file():
            missing.append(stem)
            continue
        notes = midi_file_to_notes(midi_path)
        estimates[stem] = notes
        prediction_parts.append(notes.assign(stem=stem))
    if missing:
        raise RuntimeError(f"Missing ROSVOT MIDI predictions for {missing}")

    micro, per_recording, macro = evaluate_corpus(references, estimates, stems)
    per_recording.to_csv(OUTPUT / "per_recording_metrics.csv", index=False)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions[["stem", "onset", "offset", "midi", "hz"]].to_csv(
        OUTPUT / "predicted_notes.csv.gz", index=False
    )

    checkpoint_paths = [
        ROSVOT_ROOT / "checkpoints" / "rmvpe" / "model.pt",
        ROSVOT_ROOT / "checkpoints" / "rosvot" / "config.yaml",
        ROSVOT_ROOT / "checkpoints" / "rosvot" / "model.pt",
        ROSVOT_ROOT / "checkpoints" / "rwbd" / "config.yaml",
        ROSVOT_ROOT / "checkpoints" / "rwbd" / "model.pt",
    ]
    for path in checkpoint_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = {
        "model": "ROSVOT official pretrained checkpoint",
        "code_source": "https://github.com/RickyL-2000/ROSVOT",
        "evaluation_corpus": "MTG-QBH/Molina corrected note annotations",
        "recordings": len(stems),
        "reference_notes": int(sum(len(notes) for notes in references.values())),
        "estimated_notes": int(sum(len(notes) for notes in estimates.values())),
        "inference": {
            "word_boundaries": "official pretrained RWBD predictor",
            "note_boundary_threshold": NOTE_BOUNDARY_THRESHOLD,
            "threshold_source": "official inference CLI default",
            "test_recordings_used_for_selection": False,
            "time_shift_seconds": 0.0,
        },
        "standard_molina_micro": micro,
        "standard_molina_macro": macro,
        "inference_environment": inference_environment(),
        "archive_sha256": file_sha256(ARCHIVE),
        "checkpoint_sha256": {
            str(path.relative_to(ROSVOT_ROOT)): file_sha256(path) for path in checkpoint_paths
        },
        "artifacts": {
            "manifest": str(manifest_path.relative_to(ROOT)),
            "raw_inference": str(INFERENCE.relative_to(ROOT)),
            "per_recording_metrics": "revision/results/rosvot_molina/per_recording_metrics.csv",
            "predicted_notes": "revision/results/rosvot_molina/predicted_notes.csv.gz",
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    report = [
        "# Official ROSVOT on the 38 Molina recordings",
        "",
        "The official M4Singer-trained ROSVOT, RWBD, and RMVPE checkpoints were run",
        "directly on the same 38 waveforms used by the revised Molina evaluation.",
        f"The official inference default note-boundary threshold ({NOTE_BOUNDARY_THRESHOLD:.2f})",
        "was retained. No Molina annotation, test-derived time shift, or threshold tuning",
        "was used for ROSVOT inference.",
        "",
        "| Aggregation | COnPOff | COnP | COn | Reference notes | Estimated notes |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Micro | {micro['COnPOff_F']:.4f} | {micro['COnP_F']:.4f} | "
            f"{micro['COn_F']:.4f} | {micro['ref_notes']} | {micro['est_notes']} |"
        ),
        (
            f"| Macro | {macro['COnPOff_F']:.4f} | {macro['COnP_F']:.4f} | "
            f"{macro['COn_F']:.4f} | -- | -- |"
        ),
        "",
        "These are the standard Molina metrics used for the proposed method, not the",
        "special onset-only, octave-invariant HumTrans metric.",
    ]
    (OUTPUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
