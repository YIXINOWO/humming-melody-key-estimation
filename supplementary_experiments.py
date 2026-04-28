"""
Supplementary experiments for EI journal submission.

Reuses pre-computed artifacts:
- final_results_for_paper/frame_onset_offset_oof.csv  (frame-level features + GT labels)
- final_results_for_paper/all_predictions_onoff.csv   (baseline note predictions)
- final_results_for_paper/overall_metrics_onoff.csv   (best hyperparameters)
- gt_files_temp/<stem>.GroundTruth.txt                (38 GT files)

Adds five experiments:
  E1. Ablation study (5 feature group configurations)
  E2. Heuristic (CREPE pitch+energy, non-ML) baseline
  E3. Split/Merged post-processing improvement
  E4. Bootstrap significance test (95% CI on COnPOff/COnP/COn)
  E5. Key-aware pitch snapping (couples key estimation into note pitch)

Outputs written to final_results_for_paper/.

Usage on server (recommended):
    # 选项 A: 一行命令，脚本自己装依赖（用清华镜像）
    python supplementary_experiments.py --auto-install

    # 选项 B: 手动装依赖再跑（更可控）
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \\
        numpy pandas scipy scikit-learn matplotlib mir_eval
    python supplementary_experiments.py

    # 试跑（约 3x 快，用于先验证流程跑通）
    python supplementary_experiments.py --quick

    # 自定义镜像
    python supplementary_experiments.py --auto-install \\
        --mirror https://mirrors.aliyun.com/pypi/simple
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

# ---- early argv (parse before imports so --auto-install works) ----
_early = argparse.ArgumentParser(add_help=False)
_early.add_argument("--auto-install", action="store_true",
                    help="auto pip install missing deps using a Chinese mirror")
_early.add_argument("--mirror", type=str, default="https://pypi.tuna.tsinghua.edu.cn/simple",
                    help="pip index URL (default: TUNA / Tsinghua)")
_early_args, _ = _early.parse_known_args()


def _pip_install(pkgs: list[str], mirror: str) -> bool:
    cmd = [sys.executable, "-m", "pip", "install",
           "-i", mirror,
           "--trusted-host", mirror.split("//")[-1].split("/")[0],
           "--upgrade", *pkgs]
    print(f"[auto-install] running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[auto-install] FAILED: {e}", file=sys.stderr)
        return False


def _ensure(pkg: str, import_name: str | None = None) -> bool:
    name = import_name or pkg
    try:
        __import__(name)
        return True
    except ImportError:
        if not _early_args.auto_install:
            return False
        print(f"[auto-install] {pkg} missing, installing...")
        if not _pip_install([pkg], _early_args.mirror):
            return False
        try:
            __import__(name)
            return True
        except ImportError:
            return False


# ---- dependency check (with optional auto-install) ----
_deps = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("matplotlib", "matplotlib"),
    ("mir_eval", "mir_eval"),
]
_missing = [pkg for pkg, mod in _deps if not _ensure(pkg, mod)]
if _missing:
    print(f"[FATAL] missing packages: {_missing}\n"
          f"Quick fix:\n"
          f"  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple {' '.join(_missing)}\n"
          f"Or rerun this script with --auto-install:\n"
          f"  python {Path(__file__).name} --auto-install",
          file=sys.stderr)
    sys.exit(1)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mir_eval
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

# ---- full argv ----
parser = argparse.ArgumentParser()
parser.add_argument("--auto-install", action="store_true",
                    help="auto pip install missing deps using a Chinese mirror")
parser.add_argument("--mirror", type=str, default="https://pypi.tuna.tsinghua.edu.cn/simple",
                    help="pip index URL (default: TUNA / Tsinghua)")
parser.add_argument("--quick", action="store_true",
                    help="use smaller RF (n_estimators=120) for faster turnaround")
parser.add_argument("--n-boot", type=int, default=2000,
                    help="bootstrap iterations for 95%% CI (default 2000)")
parser.add_argument("--n-perm", type=int, default=5000,
                    help="paired bootstrap iterations for p-value (default 5000)")
ARGS = parser.parse_args()

RF_N_EST_ABLATION = 120 if ARGS.quick else 200
print(f"[CFG] mode={'QUICK' if ARGS.quick else 'FULL'}  "
      f"n_estimators(ablation)={RF_N_EST_ABLATION}  "
      f"n_boot={ARGS.n_boot}  n_perm={ARGS.n_perm}")
T0 = time.time()

ROOT = Path(__file__).resolve().parent
GT_DIR = ROOT / "gt_files_temp"
OUT_DIR = ROOT / "final_results_for_paper"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR  # keep a flat layout

# ------- evaluation config (same as notebook) -------
ONSET_TOL = 0.050
PITCH_TOL_CENTS = 50.0
OFFSET_RATIO = 0.20
OFFSET_MIN_TOL = 0.050
HOP = 0.010

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
MAJOR_SCALE = np.array([0, 2, 4, 5, 7, 9, 11])
MINOR_SCALE = np.array([0, 2, 3, 5, 7, 8, 10])


def midi_to_hz(m):
    return 440.0 * (2.0 ** ((np.asarray(m, dtype=float) - 69.0) / 12.0))


def read_ground_truth(path: Path) -> pd.DataFrame:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    df = pd.DataFrame(data[:, :3], columns=["onset", "offset", "midi"])
    df["hz"] = midi_to_hz(df["midi"].to_numpy())
    return df


# ------- segmentation error rates (Molina-style approximation) -------
def segmentation_error_rates(ref_intervals, est_intervals):
    n_ref = max(len(ref_intervals), 1)
    if len(est_intervals) == 0:
        return 0.0, 0.0, 0.0
    overlap = np.zeros((len(ref_intervals), len(est_intervals)), dtype=bool)
    for i, (ro, rf) in enumerate(ref_intervals):
        for j, (eo, ef) in enumerate(est_intervals):
            overlap[i, j] = min(rf, ef) > max(ro, eo)
    split = np.sum(overlap.sum(axis=1) > 1) / n_ref
    merged = np.sum(overlap.sum(axis=0) > 1) / n_ref
    spurious = np.sum(overlap.sum(axis=0) == 0) / n_ref
    return float(split), float(merged), float(spurious)


def evaluate_pred(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    """Identical protocol to notebook (mir_eval transcription + Molina seg errors)."""
    ref_intervals = gt_df[["onset", "offset"]].to_numpy(dtype=float)
    ref_hz = gt_df["hz"].to_numpy(dtype=float)
    if len(pred_df):
        est_intervals = pred_df[["onset", "offset"]].to_numpy(dtype=float)
        est_hz = pred_df["hz"].to_numpy(dtype=float)
    else:
        est_intervals = np.empty((0, 2))
        est_hz = np.empty((0,))

    if len(ref_intervals) and len(est_intervals):
        cpo_p, cpo_r, cpo_f, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_hz, est_intervals, est_hz,
            onset_tolerance=ONSET_TOL, pitch_tolerance=PITCH_TOL_CENTS,
            offset_ratio=OFFSET_RATIO, offset_min_tolerance=OFFSET_MIN_TOL,
        )
        cp_p, cp_r, cp_f, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_hz, est_intervals, est_hz,
            onset_tolerance=ONSET_TOL, pitch_tolerance=PITCH_TOL_CENTS,
            offset_ratio=None,
        )
        c_p, c_r, c_f = mir_eval.transcription.onset_precision_recall_f1(
            ref_intervals, est_intervals, onset_tolerance=ONSET_TOL,
        )
    else:
        cpo_p = cpo_r = cpo_f = cp_p = cp_r = cp_f = c_p = c_r = c_f = 0.0
    split, merged, spurious = segmentation_error_rates(ref_intervals, est_intervals)
    return {
        "ref_notes": int(len(ref_intervals)),
        "est_notes": int(len(est_intervals)),
        "COnPOff_P": cpo_p, "COnPOff_R": cpo_r, "COnPOff_F": cpo_f,
        "COnP_F": cp_f, "COn_F": c_f,
        "Split": split, "Merged": merged, "Spurious": spurious,
    }


def evaluate_per_stem(gt_by_stem: dict, pred_by_stem: dict) -> pd.DataFrame:
    rows = []
    for stem, gt in gt_by_stem.items():
        pred = pred_by_stem.get(stem, pd.DataFrame(columns=["onset", "offset", "midi", "hz"]))
        m = evaluate_pred(gt, pred)
        rows.append({"stem": stem, **m})
    return pd.DataFrame(rows)


def evaluate_concat(gt_by_stem: dict, pred_by_stem: dict) -> dict:
    """Shift each recording to a non-overlapping global timeline, then evaluate once."""
    gt_all, pred_all = [], []
    shift = 0.0
    for stem in gt_by_stem.keys():
        gt = gt_by_stem[stem].copy()
        pred = pred_by_stem.get(stem, pd.DataFrame(columns=["onset", "offset", "midi", "hz"])).copy()
        gt[["onset", "offset"]] += shift
        if len(pred):
            pred[["onset", "offset"]] += shift
        gt_all.append(gt)
        pred_all.append(pred)
        last_gt = float(gt["offset"].max()) if len(gt) else 0.0
        last_pred = float(pred["offset"].max()) if len(pred) else 0.0
        shift = max(last_gt, last_pred) + 10.0
    gt_concat = pd.concat(gt_all, ignore_index=True)
    pred_concat = pd.concat(pred_all, ignore_index=True) if pred_all else pd.DataFrame(columns=["onset", "offset", "midi", "hz"])
    return evaluate_pred(gt_concat, pred_concat)


# ------- shared utilities for note generation from OOF probas -------
def fill_short_bool_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = mask.copy().astype(bool)
    n = len(out)
    i = 0
    while i < n:
        if out[i]:
            i += 1
            continue
        j = i
        while j < n and not out[j]:
            j += 1
        if (j - i) <= max_gap and i > 0 and j < n and out[i - 1] and out[j]:
            out[i:j] = True
        i = j
    return out


def contiguous_regions(mask: np.ndarray):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(changes[i]), int(changes[i + 1])) for i in range(0, len(changes), 2)]


def pick_peaks(times: np.ndarray, scores: np.ndarray, threshold: float, min_sep: float) -> np.ndarray:
    cand = []
    for i in range(1, len(scores) - 1):
        if scores[i] >= threshold and scores[i] >= scores[i - 1] and scores[i] >= scores[i + 1]:
            cand.append(i)
    cand = sorted(cand, key=lambda idx: scores[idx], reverse=True)
    selected = []
    for idx in cand:
        if all(abs(times[idx] - times[j]) >= min_sep for j in selected):
            selected.append(idx)
    return np.array(sorted(selected), dtype=int)


def notes_from_probas(stem_part: pd.DataFrame, on_th: float, off_th: float,
                      min_sep: float, min_dur: float) -> pd.DataFrame:
    times = stem_part["time"].to_numpy(dtype=float)
    midi = stem_part["midi"].to_numpy(dtype=float)
    conf = stem_part["conf"].to_numpy(dtype=float)
    voiced = (stem_part["voiced"].to_numpy(dtype=float) > 0.5) & (conf >= 0.25)
    voiced = fill_short_bool_gaps(voiced, 3)
    on_p = stem_part["onset_proba"].to_numpy(dtype=float)
    off_p = stem_part["offset_proba"].to_numpy(dtype=float)
    onset_idx = pick_peaks(times, on_p, on_th, min_sep)
    offset_idx = pick_peaks(times, off_p, off_th, min_sep * 0.6)

    notes = []
    for run_start, run_end in contiguous_regions(voiced):
        run_onsets = [int(i) for i in onset_idx if run_start <= i < run_end]
        run_offsets = [int(i) for i in offset_idx if run_start < i <= run_end]
        if not run_onsets:
            run_onsets = [run_start]
        if run_onsets[0] - run_start > int(round(0.120 / HOP)):
            run_onsets = [run_start] + run_onsets
        run_onsets = sorted(set(run_onsets))
        for j, onset_frame in enumerate(run_onsets):
            next_onset = run_onsets[j + 1] if j + 1 < len(run_onsets) else run_end
            min_off = onset_frame + max(1, int(round(min_dur / HOP)))
            max_off = min(next_onset + int(round(0.120 / HOP)), run_end)
            cands = [k for k in run_offsets if min_off <= k <= max_off]
            if cands:
                before_next = [k for k in cands if k <= next_onset]
                offset_frame = before_next[-1] if before_next else max(cands, key=lambda k: off_p[k])
            else:
                offset_frame = next_onset
            if offset_frame <= onset_frame:
                continue
            on_t = float(times[max(0, min(onset_frame, len(times) - 1))])
            off_t = float(times[max(0, min(offset_frame, len(times) - 1))] + HOP)
            if off_t - on_t < min_dur:
                continue
            vals = midi[onset_frame:offset_frame]
            vals = vals[np.isfinite(vals)]
            if len(vals) < 2:
                continue
            raw_midi = float(np.nanmedian(vals))
            note_midi = float(np.round(raw_midi))  # nearest semitone
            notes.append({"onset": on_t, "offset": off_t, "midi": note_midi,
                          "hz": float(midi_to_hz(note_midi)), "raw_midi": raw_midi})
    return pd.DataFrame(notes)


# ------- key estimation -------
def estimate_key(midi: np.ndarray, weights=None, conf=None):
    valid = np.isfinite(midi)
    if conf is not None:
        thr = np.nanpercentile(conf[np.isfinite(conf)], 55) if np.isfinite(conf).any() else 0.0
        valid &= conf >= thr
    vals = midi[valid]
    if len(vals) < 5:
        return {"tonic": 0, "mode": "major", "name": "C major"}
    if weights is None:
        w = np.ones(len(vals))
    else:
        w = np.asarray(weights, dtype=float)[valid]
        w = np.where(np.isfinite(w) & (w > 0), w, 0.01)
    residual = vals - np.round(vals)
    residual = ((residual + 0.5) % 1.0) - 0.5
    order = np.argsort(residual)
    cum = np.cumsum(w[order]) / (np.sum(w) + 1e-12)
    tuning = float(residual[order][np.searchsorted(cum, 0.5)])
    pcs = np.mod(np.round(vals - tuning).astype(int), 12)
    hist = np.zeros(12)
    for pc, ww in zip(pcs, w):
        hist[int(pc)] += max(float(ww), 0.01)
    hist = hist / (hist.sum() + 1e-12)
    best = None
    for mode, profile in [("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)]:
        prof = profile / profile.sum()
        for tonic in range(12):
            shifted = np.roll(prof, tonic)
            score = float(np.dot(hist, shifted) / (np.linalg.norm(hist) * np.linalg.norm(shifted) + 1e-12))
            if best is None or score > best["score"]:
                best = {"tonic": tonic, "mode": mode, "name": f"{NOTE_NAMES[tonic]} {mode}", "score": score}
    return best


def key_pcs(key_info: dict) -> set:
    scale = MAJOR_SCALE if key_info["mode"] == "major" else MINOR_SCALE
    return set(int((key_info["tonic"] + d) % 12) for d in scale)


def snap_to_key(raw_midi: float, key_info: dict, max_snap: float = 0.75) -> float:
    if not np.isfinite(raw_midi):
        return raw_midi
    nearest_int = int(np.round(raw_midi))
    pcs = key_pcs(key_info)
    if (nearest_int % 12) in pcs:
        return float(nearest_int)
    candidates = []
    for d in range(-2, 3):
        cand = nearest_int + d
        if (cand % 12) in pcs and abs(cand - raw_midi) <= max_snap:
            candidates.append(cand)
    if not candidates:
        return float(nearest_int)
    return float(min(candidates, key=lambda c: abs(c - raw_midi)))


# =====================================================================
# Load shared resources
# =====================================================================
print("Loading resources...")
required_files = [
    OUT_DIR / "frame_onset_offset_oof.csv",
    OUT_DIR / "all_predictions_onoff.csv",
    OUT_DIR / "overall_metrics_onoff.csv",
]
for p in required_files:
    if not p.exists():
        print(f"[FATAL] missing required file: {p}\n"
              f"Please run music.ipynb first to generate it.", file=sys.stderr)
        sys.exit(2)
if not GT_DIR.exists():
    print(f"[FATAL] missing GT directory: {GT_DIR}", file=sys.stderr)
    sys.exit(2)

frame_data = pd.read_csv(OUT_DIR / "frame_onset_offset_oof.csv")
print(f"  frame_data: {len(frame_data):,} rows, {frame_data['stem'].nunique()} stems")
all_pred = pd.read_csv(OUT_DIR / "all_predictions_onoff.csv")
print(f"  baseline predictions: {len(all_pred):,} notes")

# Best hyperparameters from overall_metrics_onoff.csv
overall = pd.read_csv(OUT_DIR / "overall_metrics_onoff.csv").iloc[0]
BEST_ON_TH = float(overall["onset_threshold"])
BEST_OFF_TH = float(overall["offset_threshold"])
BEST_MIN_SEP = float(overall["min_sep"])
BEST_MIN_DUR = float(overall["min_duration"])
print(f"  best params: on_th={BEST_ON_TH}, off_th={BEST_OFF_TH}, min_sep={BEST_MIN_SEP}, min_dur={BEST_MIN_DUR}")

eval_stems = sorted(frame_data["stem"].unique().tolist())
print(f"  evaluation stems: {len(eval_stems)}")

gt_by_stem = {s: read_ground_truth(GT_DIR / f"{s}.GroundTruth.txt") for s in eval_stems}
print(f"  GT loaded for {len(gt_by_stem)} stems")

baseline_pred_by_stem = {s: g.drop(columns=["stem"]).reset_index(drop=True)
                        for s, g in all_pred.groupby("stem")}
for s in eval_stems:
    baseline_pred_by_stem.setdefault(s, pd.DataFrame(columns=["onset", "offset", "midi", "hz"]))


# =====================================================================
# E1. Ablation study
# =====================================================================
print("\n" + "=" * 70)
print("E1. Ablation study")
print("=" * 70)

# Define 5 nested feature configurations
ABLATION_CONFIGS = [
    ("A1: F0 only",
     ["midi", "conf", "voiced", "dm", "abs_dm", "ddm"]),
    ("A2: + RMS energy",
     ["midi", "conf", "voiced", "dm", "abs_dm", "ddm",
      "rms", "drms"]),
    ("A3: + spectral flux",
     ["midi", "conf", "voiced", "dm", "abs_dm", "ddm",
      "rms", "drms", "flux", "dflux"]),
    ("A4: + voicing-run / pitch-std",
     ["midi", "conf", "voiced", "dm", "abs_dm", "ddm",
      "rms", "drms", "flux", "dflux",
      "local_pitch_std", "voice_since_start", "voice_to_end", "voice_run_pos"]),
]
# A5 (full) = A4 + lag features
LAG_FEATS = [c for c in frame_data.columns if "_lag" in c]
ABLATION_CONFIGS.append(("A5: + temporal lags (full)",
                         ABLATION_CONFIGS[-1][1] + LAG_FEATS))

groups = frame_data["stem"].to_numpy()


def fit_oof(label_col: str, feat_cols: list[str], seed_offset: int = 0) -> np.ndarray:
    X = frame_data[feat_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(dtype=float)
    y = frame_data[label_col].to_numpy(dtype=int)
    out = np.zeros(len(frame_data), dtype=float)
    gkf = GroupKFold(n_splits=5)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        clf = RandomForestClassifier(
            n_estimators=RF_N_EST_ABLATION, max_depth=14, min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=2026 + fold + seed_offset, n_jobs=-1,
        )
        clf.fit(X[tr], y[tr])
        out[te] = clf.predict_proba(X[te])[:, 1]
    return out


ablation_rows = []
for name, feats in ABLATION_CONFIGS:
    print(f"  fitting {name}  ({len(feats)} features)...")
    on_p = fit_oof("onset_label", feats, seed_offset=0)
    off_p = fit_oof("offset_label", feats, seed_offset=50)
    fd = frame_data[["stem", "time", "midi", "conf", "voiced"]].copy()
    fd["onset_proba"] = on_p
    fd["offset_proba"] = off_p

    pred_by_stem = {}
    for stem in eval_stems:
        part = fd[fd["stem"] == stem].reset_index(drop=True)
        pred_by_stem[stem] = notes_from_probas(part, BEST_ON_TH, BEST_OFF_TH,
                                               BEST_MIN_SEP, BEST_MIN_DUR)
    m = evaluate_concat(gt_by_stem, pred_by_stem)
    row = {"Config": name, "n_features": len(feats), **m}
    ablation_rows.append(row)
    print(f"    -> COnPOff={m['COnPOff_F']:.4f}  COnP={m['COnP_F']:.4f}  COn={m['COn_F']:.4f}")

ablation_df = pd.DataFrame(ablation_rows)
ablation_df.to_csv(OUT_DIR / "ablation_study.csv", index=False)
print(f"  saved -> ablation_study.csv")

# Plot ablation bar chart
fig, ax = plt.subplots(figsize=(9, 4.6))
x = np.arange(len(ablation_df))
width = 0.25
ax.bar(x - width, ablation_df["COnPOff_F"], width, label="COnPOff", color="#3a7ca5")
ax.bar(x,         ablation_df["COnP_F"],   width, label="COnP",    color="#81c3d7")
ax.bar(x + width, ablation_df["COn_F"],    width, label="COn",     color="#16425b")
ax.set_xticks(x)
ax.set_xticklabels([c.split(":")[0] for c in ablation_df["Config"]], fontsize=10)
ax.set_ylim(0, 1.0)
ax.set_ylabel("F-measure", fontsize=11)
ax.set_xlabel("Feature configuration (cumulative)", fontsize=11)
ax.set_title("Ablation study: feature group contribution", fontsize=12)
for i, v in enumerate(ablation_df["COnPOff_F"]):
    ax.text(i - width, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "ablation_study_bar.png", dpi=200)
plt.close()
print("  saved -> ablation_study_bar.png")


# =====================================================================
# E2. Heuristic (non-ML) baseline
#     A simple but fair baseline that does NOT use the trained RF:
#     uses CREPE pitch + voicing + RMS valleys for onset, semitone change for split.
# =====================================================================
print("\n" + "=" * 70)
print("E2. Heuristic non-ML baseline")
print("=" * 70)


def heuristic_notes(stem_part: pd.DataFrame, min_dur: float = 0.10) -> pd.DataFrame:
    """Pitch+energy heuristic: voicing runs, then split inside a run wherever
       (a) integer semitone of smoothed MIDI changes, OR
       (b) RMS local minimum > 30% drop."""
    times = stem_part["time"].to_numpy(dtype=float)
    midi = stem_part["midi"].to_numpy(dtype=float)
    conf = stem_part["conf"].to_numpy(dtype=float)
    rms = stem_part["rms"].to_numpy(dtype=float)
    voiced = (stem_part["voiced"].to_numpy(dtype=float) > 0.5) & (conf >= 0.25)
    voiced = fill_short_bool_gaps(voiced, 3)
    notes = []
    for run_start, run_end in contiguous_regions(voiced):
        run_midi = midi[run_start:run_end]
        run_rms = rms[run_start:run_end]
        if len(run_midi) < 4:
            continue
        # Boundary: semitone change of integer rounded MIDI
        floors = np.round(run_midi).astype(int)
        change = np.where(np.diff(floors) != 0)[0] + 1
        # Boundary: RMS valley with >=30% drop
        rms_smooth = pd.Series(run_rms).rolling(5, center=True, min_periods=1).mean().to_numpy()
        valleys = []
        for k in range(1, len(rms_smooth) - 1):
            if rms_smooth[k] < rms_smooth[k - 1] and rms_smooth[k] < rms_smooth[k + 1]:
                left_max = np.max(rms_smooth[max(0, k - 6):k + 1])
                right_max = np.max(rms_smooth[k:min(len(rms_smooth), k + 7)])
                if rms_smooth[k] < 0.7 * min(left_max, right_max):
                    valleys.append(k)
        boundaries = sorted(set([0] + list(change) + valleys + [len(run_midi)]))
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            if b - a < int(round(min_dur / HOP)):
                continue
            seg = run_midi[a:b]
            seg = seg[np.isfinite(seg)]
            if len(seg) < 2:
                continue
            raw = float(np.nanmedian(seg))
            note_m = float(np.round(raw))
            on_t = float(times[run_start + a])
            off_t = float(times[run_start + min(b, len(run_midi) - 1)] + HOP)
            notes.append({"onset": on_t, "offset": off_t, "midi": note_m,
                          "hz": float(midi_to_hz(note_m)), "raw_midi": raw})
    return pd.DataFrame(notes)


heur_pred_by_stem = {}
for stem in eval_stems:
    part = frame_data[frame_data["stem"] == stem].reset_index(drop=True)
    heur_pred_by_stem[stem] = heuristic_notes(part)

heur_metrics = evaluate_concat(gt_by_stem, heur_pred_by_stem)
print(f"  Heuristic: COnPOff={heur_metrics['COnPOff_F']:.4f}  COnP={heur_metrics['COnP_F']:.4f}  COn={heur_metrics['COn_F']:.4f}")

baseline_metrics = evaluate_concat(gt_by_stem, baseline_pred_by_stem)
print(f"  Ours-OnOff: COnPOff={baseline_metrics['COnPOff_F']:.4f}")

# Stash for later use
ours_pred_by_stem = baseline_pred_by_stem


# =====================================================================
# E3. Split/Merged post-processing
# =====================================================================
print("\n" + "=" * 70)
print("E3. Split/Merged post-processing")
print("=" * 70)


def post_process_notes(notes: pd.DataFrame,
                       merge_gap_sec: float = 0.060,
                       merge_pitch_tol: float = 0.55,
                       min_dur_keep: float = 0.080) -> pd.DataFrame:
    """1) merge adjacent notes if same nearest-semitone & gap < merge_gap_sec
       2) drop notes shorter than min_dur_keep that are surrounded by longer notes."""
    if len(notes) == 0:
        return notes.copy()
    df = notes.sort_values("onset").reset_index(drop=True).copy()
    # Step 1: merge
    merged_rows = [df.iloc[0].to_dict()]
    for i in range(1, len(df)):
        cur = df.iloc[i].to_dict()
        prev = merged_rows[-1]
        gap = cur["onset"] - prev["offset"]
        same_pitch = abs(cur["midi"] - prev["midi"]) <= merge_pitch_tol
        if gap <= merge_gap_sec and same_pitch:
            prev["offset"] = cur["offset"]
            # weighted-by-duration midi update
            d_prev = max(prev["offset"] - prev["onset"], 1e-3)
            d_cur = max(cur["offset"] - cur["onset"], 1e-3)
            prev["midi"] = (prev["midi"] * d_prev + cur["midi"] * d_cur) / (d_prev + d_cur)
            prev["hz"] = float(midi_to_hz(prev["midi"]))
            if "raw_midi" in cur:
                prev["raw_midi"] = (prev.get("raw_midi", prev["midi"]) * d_prev +
                                    cur["raw_midi"] * d_cur) / (d_prev + d_cur)
        else:
            merged_rows.append(cur)
    out = pd.DataFrame(merged_rows)
    # Step 2: drop short isolated notes
    keep = np.ones(len(out), dtype=bool)
    durs = (out["offset"] - out["onset"]).to_numpy(dtype=float)
    for i in range(len(out)):
        if durs[i] < min_dur_keep:
            left_ok = i > 0 and durs[i - 1] >= min_dur_keep
            right_ok = i < len(out) - 1 and durs[i + 1] >= min_dur_keep
            if left_ok and right_ok:
                keep[i] = False
    return out.loc[keep].reset_index(drop=True)


post_pred_by_stem = {s: post_process_notes(p) for s, p in baseline_pred_by_stem.items()}
post_metrics = evaluate_concat(gt_by_stem, post_pred_by_stem)
print(f"  Pre-postproc:  COnPOff={baseline_metrics['COnPOff_F']:.4f}  Split={baseline_metrics['Split']:.3f}  Merged={baseline_metrics['Merged']:.3f}  Spurious={baseline_metrics['Spurious']:.3f}")
print(f"  Post-postproc: COnPOff={post_metrics['COnPOff_F']:.4f}  Split={post_metrics['Split']:.3f}  Merged={post_metrics['Merged']:.3f}  Spurious={post_metrics['Spurious']:.3f}")


# =====================================================================
# E5. Key-aware pitch snapping (run before E4 so we have the final variant)
# =====================================================================
print("\n" + "=" * 70)
print("E5. Key-aware pitch snapping")
print("=" * 70)


def per_stem_key(stem: str) -> dict:
    part = frame_data[frame_data["stem"] == stem]
    midi = part["midi"].to_numpy(dtype=float)
    conf = part["conf"].to_numpy(dtype=float)
    return estimate_key(midi, weights=conf, conf=conf)


keys_per_stem = {s: per_stem_key(s) for s in eval_stems}
print(f"  estimated keys for {len(keys_per_stem)} stems")


def apply_key_snap(notes: pd.DataFrame, key_info: dict) -> pd.DataFrame:
    if len(notes) == 0:
        return notes
    out = notes.copy()
    raw = out["raw_midi"].to_numpy(dtype=float) if "raw_midi" in out.columns else out["midi"].to_numpy(dtype=float)
    snapped = np.array([snap_to_key(float(v), key_info) for v in raw])
    out["midi"] = snapped
    out["hz"] = midi_to_hz(snapped)
    return out


# Apply snap to baseline
key_snap_pred_by_stem = {s: apply_key_snap(p, keys_per_stem[s]) for s, p in baseline_pred_by_stem.items()}
key_snap_metrics = evaluate_concat(gt_by_stem, key_snap_pred_by_stem)
print(f"  Without key snap: COnPOff={baseline_metrics['COnPOff_F']:.4f}  COnP={baseline_metrics['COnP_F']:.4f}")
print(f"  With key snap:    COnPOff={key_snap_metrics['COnPOff_F']:.4f}  COnP={key_snap_metrics['COnP_F']:.4f}")

# Combined: post-process + key snap
combined_pred_by_stem = {s: apply_key_snap(post_pred_by_stem[s], keys_per_stem[s]) for s in eval_stems}
combined_metrics = evaluate_concat(gt_by_stem, combined_pred_by_stem)
print(f"  Postproc + key snap: COnPOff={combined_metrics['COnPOff_F']:.4f}  COnP={combined_metrics['COnP_F']:.4f}  Split={combined_metrics['Split']:.3f}  Merged={combined_metrics['Merged']:.3f}")


# Save Split/Merged comparison table (covers E3 + E5 results)
postproc_rows = [
    {"Variant": "Ours-OnOff (baseline)",         **baseline_metrics},
    {"Variant": "+ post-processing",             **post_metrics},
    {"Variant": "+ key-aware snap",              **key_snap_metrics},
    {"Variant": "+ post-processing + key snap",  **combined_metrics},
]
postproc_df = pd.DataFrame(postproc_rows)
postproc_df.to_csv(OUT_DIR / "postprocess_keysnap_ablation.csv", index=False)
print("  saved -> postprocess_keysnap_ablation.csv")

# Plot Split/Merged/Spurious before/after
fig, ax = plt.subplots(figsize=(9, 4.4))
labels = ["COnPOff", "COnP", "COn", "Split", "Merged", "Spurious"]
keys = ["COnPOff_F", "COnP_F", "COn_F", "Split", "Merged", "Spurious"]
colors_v = ["#3a7ca5", "#81c3d7", "#16425b", "#d62828"]
x = np.arange(len(labels))
w = 0.20
for i, row in enumerate(postproc_rows):
    ax.bar(x + (i - 1.5) * w, [row[k] for k in keys], w,
           label=row["Variant"], color=colors_v[i])
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Score / error rate", fontsize=11)
ax.set_title("Effect of post-processing and key-aware pitch snapping", fontsize=12)
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.legend(loc="upper right", fontsize=8.5)
plt.tight_layout()
plt.savefig(FIG_DIR / "postproc_keysnap_bar.png", dpi=200)
plt.close()
print("  saved -> postproc_keysnap_bar.png")


# =====================================================================
# E2 (cont.) – build the comprehensive method comparison table
# =====================================================================
print("\n" + "=" * 70)
print("E2 (final). Method comparison table")
print("=" * 70)
method_rows = [
    {"Methods": "Li et al. 2021 Steps 1+2",   "COnPOff": 0.525, "COnP": 0.712, "COn": 0.761,
     "Split": 0.013, "Merged": 0.235, "Spurious": 0.128},
    {"Methods": "Li et al. 2021 Steps 1+3",   "COnPOff": 0.520, "COnP": 0.683, "COn": 0.741,
     "Split": 0.079, "Merged": 0.233, "Spurious": 0.114},
    {"Methods": "Li et al. 2021 Steps 1+2+3", "COnPOff": 0.610, "COnP": 0.762, "COn": 0.807,
     "Split": 0.093, "Merged": 0.078, "Spurious": 0.035},
    {"Methods": "Heuristic (CREPE + RMS, no ML)",
     "COnPOff": heur_metrics["COnPOff_F"], "COnP": heur_metrics["COnP_F"], "COn": heur_metrics["COn_F"],
     "Split": heur_metrics["Split"], "Merged": heur_metrics["Merged"], "Spurious": heur_metrics["Spurious"]},
    {"Methods": "Ours-OnOff",
     "COnPOff": baseline_metrics["COnPOff_F"], "COnP": baseline_metrics["COnP_F"], "COn": baseline_metrics["COn_F"],
     "Split": baseline_metrics["Split"], "Merged": baseline_metrics["Merged"], "Spurious": baseline_metrics["Spurious"]},
    {"Methods": "Ours-OnOff + post-proc",
     "COnPOff": post_metrics["COnPOff_F"], "COnP": post_metrics["COnP_F"], "COn": post_metrics["COn_F"],
     "Split": post_metrics["Split"], "Merged": post_metrics["Merged"], "Spurious": post_metrics["Spurious"]},
    {"Methods": "Ours-Full (post-proc + key snap)",
     "COnPOff": combined_metrics["COnPOff_F"], "COnP": combined_metrics["COnP_F"], "COn": combined_metrics["COn_F"],
     "Split": combined_metrics["Split"], "Merged": combined_metrics["Merged"], "Spurious": combined_metrics["Spurious"]},
]
method_df = pd.DataFrame(method_rows)
for col in ["COnPOff", "COnP", "COn", "Split", "Merged", "Spurious"]:
    method_df[col] = method_df[col].astype(float).round(3)
method_df.to_csv(OUT_DIR / "method_comparison.csv", index=False)
print(method_df.to_string(index=False))
print("  saved -> method_comparison.csv")

# Plot grouped bar chart for the comparison
fig, ax = plt.subplots(figsize=(11, 4.8))
metrics_to_plot = ["COnPOff", "COnP", "COn"]
x = np.arange(len(method_df))
w = 0.26
colors_m = ["#3a7ca5", "#81c3d7", "#16425b"]
for i, m in enumerate(metrics_to_plot):
    ax.bar(x + (i - 1) * w, method_df[m], w, label=m, color=colors_m[i])
ax.set_xticks(x)
ax.set_xticklabels(method_df["Methods"], rotation=18, ha="right", fontsize=9)
ax.set_ylabel("F-measure", fontsize=11)
ax.set_ylim(0, 1.0)
ax.set_title("Method comparison on 38 MTG-QBH/Molina recordings", fontsize=12)
for i, v in enumerate(method_df["COnPOff"]):
    ax.text(i - w, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "method_comparison_bar.png", dpi=200)
plt.close()
print("  saved -> method_comparison_bar.png")


# =====================================================================
# E4. Bootstrap significance test (95% CI on F-measures)
# =====================================================================
print("\n" + "=" * 70)
print("E4. Bootstrap significance test")
print("=" * 70)

# Per-stem evaluation for the final model and the baseline
per_stem_baseline = evaluate_per_stem(gt_by_stem, baseline_pred_by_stem)
per_stem_combined = evaluate_per_stem(gt_by_stem, combined_pred_by_stem)
per_stem_baseline.to_csv(OUT_DIR / "per_stem_metrics_baseline.csv", index=False)
per_stem_combined.to_csv(OUT_DIR / "per_stem_metrics_full.csv", index=False)


def bootstrap_ci(per_stem_df: pd.DataFrame, n_boot: int = 2000, seed: int = 1234) -> dict:
    """Resample stems with replacement; for each bootstrap sample re-aggregate
       precision/recall by recomputing F via aggregate match counts implied per stem.
       To stay robust without recomputing matches, we use per-stem F as the unit:
       this is standard 'recording-level' bootstrap reported in MIR papers."""
    rng = np.random.default_rng(seed)
    stems = per_stem_df["stem"].to_numpy()
    cols = ["COnPOff_F", "COnP_F", "COn_F"]
    boot = {c: [] for c in cols}
    for _ in range(n_boot):
        idx = rng.integers(0, len(stems), size=len(stems))
        sample = per_stem_df.iloc[idx]
        for c in cols:
            boot[c].append(float(sample[c].mean()))
    ci = {}
    for c in cols:
        arr = np.array(boot[c])
        ci[c] = (float(np.mean(arr)),
                 float(np.percentile(arr, 2.5)),
                 float(np.percentile(arr, 97.5)))
    return ci, boot


def paired_bootstrap_pvalue(a: pd.Series, b: pd.Series, n_boot: int = 5000, seed: int = 4321) -> float:
    """Two-sided paired bootstrap p-value for mean(a-b) > 0."""
    rng = np.random.default_rng(seed)
    diff = (a.to_numpy() - b.to_numpy())
    n = len(diff)
    obs = float(np.mean(diff))
    centered = diff - obs
    count = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if abs(np.mean(centered[idx])) >= abs(obs):
            count += 1
    return (count + 1) / (n_boot + 1)


ci_baseline, boot_baseline = bootstrap_ci(per_stem_baseline, n_boot=ARGS.n_boot)
ci_full, boot_full = bootstrap_ci(per_stem_combined, n_boot=ARGS.n_boot)

bootstrap_rows = []
for c in ["COnPOff_F", "COnP_F", "COn_F"]:
    bm, bl, bh = ci_baseline[c]
    fm, fl, fh = ci_full[c]
    p = paired_bootstrap_pvalue(per_stem_combined[c], per_stem_baseline[c], n_boot=ARGS.n_perm)
    bootstrap_rows.append({
        "Metric": c.replace("_F", ""),
        "Ours-OnOff mean": bm, "Ours-OnOff 95% CI low": bl, "Ours-OnOff 95% CI high": bh,
        "Ours-Full mean": fm, "Ours-Full 95% CI low": fl, "Ours-Full 95% CI high": fh,
        "p-value (paired bootstrap)": p,
    })
bootstrap_df = pd.DataFrame(bootstrap_rows)
for col in [c for c in bootstrap_df.columns if c not in ("Metric",)]:
    bootstrap_df[col] = bootstrap_df[col].astype(float).round(4)
bootstrap_df.to_csv(OUT_DIR / "bootstrap_significance.csv", index=False)
print(bootstrap_df.to_string(index=False))
print("  saved -> bootstrap_significance.csv")

# Plot bootstrap distributions
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, c in zip(axes, ["COnPOff_F", "COnP_F", "COn_F"]):
    ax.hist(boot_baseline[c], bins=40, color="#9bbac9", alpha=0.85, label="Ours-OnOff")
    ax.hist(boot_full[c],     bins=40, color="#d68a4f", alpha=0.65, label="Ours-Full")
    bm, bl, bh = ci_baseline[c]; fm, fl, fh = ci_full[c]
    ax.axvline(bm, color="#3a7ca5", linestyle="--", linewidth=1.4)
    ax.axvline(fm, color="#a83232", linestyle="--", linewidth=1.4)
    ax.set_title(f"{c.replace('_F','')}\nbase={bm:.3f} [{bl:.3f},{bh:.3f}]\nfull={fm:.3f} [{fl:.3f},{fh:.3f}]",
                 fontsize=10)
    ax.grid(linestyle=":", alpha=0.5)
axes[0].set_ylabel("Bootstrap count", fontsize=10)
axes[-1].legend(loc="upper right", fontsize=9)
fig.suptitle("Recording-level bootstrap distribution of F-measures (n=2000)", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "bootstrap_distribution.png", dpi=200, bbox_inches="tight")
plt.close()
print("  saved -> bootstrap_distribution.png")


# =====================================================================
# Save final summary
# =====================================================================
print("\n" + "=" * 70)
print("All experiments complete.")
print("=" * 70)
summary = {
    "baseline_COnPOff": baseline_metrics["COnPOff_F"],
    "heuristic_COnPOff": heur_metrics["COnPOff_F"],
    "postproc_COnPOff": post_metrics["COnPOff_F"],
    "keysnap_COnPOff": key_snap_metrics["COnPOff_F"],
    "full_COnPOff": combined_metrics["COnPOff_F"],
    "full_Split": combined_metrics["Split"],
    "full_Merged": combined_metrics["Merged"],
    "full_Spurious": combined_metrics["Spurious"],
}
pd.DataFrame([summary]).to_csv(OUT_DIR / "supplementary_summary.csv", index=False)
print(pd.DataFrame([summary]).to_string(index=False))

elapsed = time.time() - T0
print(f"\nTotal elapsed: {elapsed/60:.1f} min")
print(f"\nAll outputs saved to: {OUT_DIR}")
print("Generated files:")
generated = [
    "ablation_study.csv", "ablation_study_bar.png",
    "method_comparison.csv", "method_comparison_bar.png",
    "postprocess_keysnap_ablation.csv", "postproc_keysnap_bar.png",
    "bootstrap_significance.csv", "bootstrap_distribution.png",
    "per_stem_metrics_baseline.csv", "per_stem_metrics_full.csv",
    "supplementary_summary.csv",
]
for fn in generated:
    p = OUT_DIR / fn
    mark = "OK " if p.exists() else "?? "
    print(f"  [{mark}] {fn}")
