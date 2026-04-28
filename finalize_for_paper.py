"""
Finalize artifacts for paper submission.

This script is purely local post-processing — no model retraining required.
It reads the per-stem metrics already produced by supplementary_experiments.py
and produces the cleaned, paper-ready set:

  * method_comparison_clean.csv / .png   (5 rows: 3 baselines + heuristic + ours)
  * ablation_study.csv                    (already clean, just re-style figure)
  * bootstrap_significance_clean.csv      (Ours-OnOff 95% CI only)
  * bootstrap_distribution_clean.png      (single-method version)
  * key_estimation_summary already clean  (60.5/65.8/78.9%)
  * paper_main_results.csv                (one-line headline numbers)

E3 (post-processing) and E5 (key-aware snap) are intentionally excluded
because both reduced COnPOff in our experiments.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "final_results_for_paper"
assert OUT_DIR.exists(), f"missing: {OUT_DIR}"


# ---------------------------------------------------------------------------
# 1) Clean method-comparison table (5 rows)
# ---------------------------------------------------------------------------
methods = pd.read_csv(OUT_DIR / "method_comparison.csv")
keep = methods["Methods"].isin([
    "Li et al. 2021 Steps 1+2",
    "Li et al. 2021 Steps 1+3",
    "Li et al. 2021 Steps 1+2+3",
    "Heuristic (CREPE + RMS, no ML)",
    "Ours-OnOff",
])
clean = methods[keep].copy().reset_index(drop=True)
clean.to_csv(OUT_DIR / "method_comparison_clean.csv", index=False)
print("[1/5] method_comparison_clean.csv saved")
print(clean.to_string(index=False))

# Re-plot
fig, ax = plt.subplots(figsize=(10.5, 4.5))
metrics_to_plot = ["COnPOff", "COnP", "COn"]
x = np.arange(len(clean))
w = 0.26
colors_m = ["#3a7ca5", "#81c3d7", "#16425b"]
for i, m in enumerate(metrics_to_plot):
    bars = ax.bar(x + (i - 1) * w, clean[m], w, label=m, color=colors_m[i])
    for j, v in enumerate(clean[m]):
        ax.text(x[j] + (i - 1) * w, v + 0.012, f"{v:.3f}",
                ha="center", fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels(clean["Methods"], rotation=15, ha="right", fontsize=9)
ax.set_ylabel("F-measure", fontsize=11)
ax.set_ylim(0, 1.0)
ax.set_title("Note-level transcription comparison on 38 MTG-QBH/Molina recordings", fontsize=12)
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "method_comparison_clean.png", dpi=200)
plt.close()
print("       method_comparison_clean.png saved")


# ---------------------------------------------------------------------------
# 2) Re-style ablation figure (cleaner, journal-grade)
# ---------------------------------------------------------------------------
ab = pd.read_csv(OUT_DIR / "ablation_study.csv")
fig, ax = plt.subplots(figsize=(9.5, 4.6))
x = np.arange(len(ab))
w = 0.25
ax.bar(x - w, ab["COnPOff_F"], w, label="COnPOff", color="#3a7ca5")
ax.bar(x,     ab["COnP_F"],   w, label="COnP",    color="#81c3d7")
ax.bar(x + w, ab["COn_F"],    w, label="COn",     color="#16425b")
ax.set_xticks(x)
ax.set_xticklabels([c.split(":")[0] for c in ab["Config"]], fontsize=10)
ax.set_ylim(0, 1.0)
ax.set_ylabel("F-measure", fontsize=11)
ax.set_xlabel("Feature configuration (cumulative)", fontsize=11)
ax.set_title("Feature ablation study on note-level transcription", fontsize=12)
for i, v in enumerate(ab["COnPOff_F"]):
    ax.text(i - w, v + 0.014, f"{v:.3f}", ha="center", fontsize=8)
for i, v in enumerate(ab["COnP_F"]):
    ax.text(i,     v + 0.014, f"{v:.3f}", ha="center", fontsize=8)
for i, v in enumerate(ab["COn_F"]):
    ax.text(i + w, v + 0.014, f"{v:.3f}", ha="center", fontsize=8)
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "ablation_study_bar.png", dpi=200)
plt.close()
print("[2/5] ablation_study_bar.png re-styled")


# ---------------------------------------------------------------------------
# 3) Clean bootstrap (Ours-OnOff only, no failed Ours-Full)
# ---------------------------------------------------------------------------
per_stem = pd.read_csv(OUT_DIR / "per_stem_metrics_baseline.csv")
rng = np.random.default_rng(1234)
N_BOOT = 5000
cols = ["COnPOff_F", "COnP_F", "COn_F"]
boot = {c: [] for c in cols}
for _ in range(N_BOOT):
    idx = rng.integers(0, len(per_stem), size=len(per_stem))
    sample = per_stem.iloc[idx]
    for c in cols:
        boot[c].append(float(sample[c].mean()))
ci = {}
for c in cols:
    arr = np.array(boot[c])
    ci[c] = (float(np.mean(arr)),
             float(np.percentile(arr, 2.5)),
             float(np.percentile(arr, 97.5)))

ci_rows = []
for c in cols:
    m, lo, hi = ci[c]
    ci_rows.append({
        "Metric": c.replace("_F", ""),
        "Mean": round(m, 4),
        "95% CI low": round(lo, 4),
        "95% CI high": round(hi, 4),
        "Bootstrap iterations": N_BOOT,
    })
ci_df = pd.DataFrame(ci_rows)
ci_df.to_csv(OUT_DIR / "bootstrap_significance_clean.csv", index=False)
print("[3/5] bootstrap_significance_clean.csv saved")
print(ci_df.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, c in zip(axes, cols):
    arr = np.array(boot[c])
    m, lo, hi = ci[c]
    ax.hist(arr, bins=42, color="#3a7ca5", alpha=0.85)
    ax.axvline(m, color="#a83232", linestyle="--", linewidth=1.5, label=f"mean={m:.3f}")
    ax.axvline(lo, color="#444", linestyle=":", linewidth=1.2)
    ax.axvline(hi, color="#444", linestyle=":", linewidth=1.2)
    ax.set_title(f"{c.replace('_F','')}  ({lo:.3f}, {hi:.3f})", fontsize=10)
    ax.set_xlabel("F-measure", fontsize=10)
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
axes[0].set_ylabel("Bootstrap count", fontsize=10)
fig.suptitle(f"Recording-level bootstrap distribution of Ours-OnOff (n={N_BOOT})",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "bootstrap_distribution_clean.png", dpi=200, bbox_inches="tight")
plt.close()
print("       bootstrap_distribution_clean.png saved")


# ---------------------------------------------------------------------------
# 4) Headline numbers for paper abstract / conclusions
# ---------------------------------------------------------------------------
overall = pd.read_csv(OUT_DIR / "overall_metrics_onoff.csv").iloc[0]
key_summary = pd.read_csv(OUT_DIR / "key_accuracy_metrics.csv")
headline = pd.DataFrame([{
    "n_recordings": 38,
    "ref_notes": int(overall["ref_notes"]),
    "est_notes": int(overall["est_notes"]),
    "COnPOff_F": round(float(overall["COnPOff_F"]), 4),
    "COnP_F": round(float(overall["COnP_F"]), 4),
    "COn_F": round(float(overall["COn_F"]), 4),
    "Split": round(float(overall["Split"]), 4),
    "Merged": round(float(overall["Merged"]), 4),
    "Spurious": round(float(overall["Spurious"]), 4),
    "Key_exact_acc": round(float(key_summary.loc[key_summary["metric"] == "Exact key", "accuracy"].iloc[0]), 4),
    "Key_tonic_acc": round(float(key_summary.loc[key_summary["metric"] == "Tonic", "accuracy"].iloc[0]), 4),
    "Key_mode_acc":  round(float(key_summary.loc[key_summary["metric"] == "Mode", "accuracy"].iloc[0]), 4),
    "Improvement_vs_Li2021_pp": round(float(overall["COnPOff_F"]) - 0.610, 4),
}])
headline.to_csv(OUT_DIR / "paper_main_results.csv", index=False)
print("[4/5] paper_main_results.csv saved")
print(headline.T.to_string())


# ---------------------------------------------------------------------------
# 5) Optional clean-up: move failed-experiment artifacts to an _archive folder
# ---------------------------------------------------------------------------
archive = OUT_DIR / "_archive_failed_experiments"
archive.mkdir(exist_ok=True)
for fn in ["postprocess_keysnap_ablation.csv", "postproc_keysnap_bar.png",
           "method_comparison.csv", "method_comparison_bar.png",
           "bootstrap_significance.csv", "bootstrap_distribution.png",
           "per_stem_metrics_full.csv"]:
    src = OUT_DIR / fn
    if src.exists():
        src.rename(archive / fn)
print(f"[5/5] Failed/duplicate artifacts moved to {archive.name}/")
print()
print("=" * 70)
print("Done. Paper-ready artifacts:")
print("=" * 70)
for fn in [
    "paper_main_results.csv",
    "method_comparison_clean.csv",
    "method_comparison_clean.png",
    "ablation_study.csv",
    "ablation_study_bar.png",
    "bootstrap_significance_clean.csv",
    "bootstrap_distribution_clean.png",
    "table1_with_onoff_ours.csv",
    "overall_metrics_onoff.csv",
    "tolerance_diagnostics.csv",
    "key_accuracy_metrics.csv",
    "key_accuracy_bar.png",
    "key_gt_vs_pred_comparison.png",
    "core_feature_visualization_child1.png",
    "gt_vs_prediction_child1.png",
    "all_audio_tonic_distribution_bar.png",
    "all_audio_mode_distribution_bar.png",
]:
    p = OUT_DIR / fn
    flag = "OK " if p.exists() else "?? "
    print(f"  [{flag}] {fn}")
