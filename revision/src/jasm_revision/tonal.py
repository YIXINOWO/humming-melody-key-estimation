"""Non-circular tonal-profile agreement utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MAJOR_PROFILE = np.asarray(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
MINOR_PROFILE = np.asarray(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)


def pitch_class_profile(midi: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Build an L1-normalized 12-bin profile from finite MIDI observations."""

    midi = np.asarray(midi, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(midi) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.zeros(12, dtype=np.float64)
    pitch_classes = np.mod(np.rint(midi[valid]).astype(np.int64), 12)
    profile = np.bincount(pitch_classes, weights=weights[valid], minlength=12).astype(
        np.float64
    )
    return profile / profile.sum()


def symbolic_profile(notes: pd.DataFrame) -> np.ndarray:
    durations = (
        notes["offset"].to_numpy(np.float64)
        - notes["onset"].to_numpy(np.float64)
    )
    return pitch_class_profile(notes["midi"].to_numpy(np.float64), durations)


def audio_profile(frames: pd.DataFrame, confidence_threshold: float) -> np.ndarray:
    confidence = frames["conf"].to_numpy(np.float64)
    voiced = frames["voiced"].to_numpy(np.float64) > 0.5
    weights = np.where(voiced & (confidence >= confidence_threshold), confidence, 0.0)
    return pitch_class_profile(frames["midi"].to_numpy(np.float64), weights)


def template_label(profile: np.ndarray) -> dict[str, int | str | float]:
    """Return a Krumhansl-Schmuckler label for descriptive comparison only."""

    if not np.isfinite(profile).all() or profile.sum() <= 0:
        raise ValueError("cannot label an empty pitch-class profile")
    best: dict[str, int | str | float] | None = None
    for mode, template in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
        normalized = template / template.sum()
        for tonic in range(12):
            shifted = np.roll(normalized, tonic)
            score = float(
                np.dot(profile, shifted)
                / (np.linalg.norm(profile) * np.linalg.norm(shifted))
            )
            if best is None or score > float(best["score"]):
                best = {
                    "tonic": tonic,
                    "mode": mode,
                    "label": f"{NOTE_NAMES[tonic]} {mode}",
                    "score": score,
                }
    assert best is not None
    return best


def profile_agreement(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float | bool]:
    if reference.sum() <= 0 or estimate.sum() <= 0:
        raise ValueError("agreement requires two non-empty profiles")
    cosine = float(
        np.dot(reference, estimate)
        / (np.linalg.norm(reference) * np.linalg.norm(estimate))
    )
    js_distance = float(jensenshannon(reference, estimate, base=2.0))
    return {
        "profile_cosine_similarity": cosine,
        "jensen_shannon_distance": js_distance,
        "dominant_pitch_class_match": bool(np.argmax(reference) == np.argmax(estimate)),
    }
