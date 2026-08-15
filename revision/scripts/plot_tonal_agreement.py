#!/usr/bin/env python3
"""Create the revised tonal-representation agreement manuscript figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "revision" / "results" / "molina_tonal_agreement"
OUTPUT_DIR = RESULT_DIR / "figure"
MANUSCRIPT_DIR = ROOT / "submission_asmp_latex_resubmission_source"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(RESULT_DIR / "per_recording_agreement.csv")
    summary = json.loads((RESULT_DIR / "summary.json").read_text(encoding="utf-8"))
    boot = {
        item["metric"]: item for item in summary["bootstrap"]
    }
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    blue = "#3973A9"
    teal = "#47A6A0"
    grey = "#D5DADF"
    dark = "#29323A"

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.35), constrained_layout=True)

    ax = axes[0]
    values = np.sort(data["profile_cosine_similarity"].to_numpy())
    ax.scatter(np.arange(1, len(values) + 1), values, s=13, color=blue, edgecolor="white", linewidth=0.35)
    mean = summary["primary_macro_metrics"]["profile_cosine_similarity"]
    ci = boot["profile_cosine_similarity"]
    ax.axhline(mean, color=dark, linewidth=1.1)
    ax.fill_between(
        [0.5, len(values) + 0.5],
        ci["bootstrap_ci_low"],
        ci["bootstrap_ci_high"],
        color=blue,
        alpha=0.16,
        linewidth=0,
    )
    ax.set(xlim=(0.5, len(values) + 0.5), ylim=(0.80, 1.005), xlabel="Recordings (sorted)", ylabel="Cosine similarity")
    ax.text(0.04, 0.08, f"mean {mean:.3f}\n95% CI {ci['bootstrap_ci_low']:.3f}–{ci['bootstrap_ci_high']:.3f}", transform=ax.transAxes, color=dark)
    ax.set_title("Direct profile similarity", fontsize=8, pad=4)
    ax.text(-0.18, 1.06, "a", transform=ax.transAxes, fontweight="bold", fontsize=9)

    ax = axes[1]
    js_values = np.sort(data["jensen_shannon_distance"].to_numpy())
    ax.scatter(np.arange(1, len(js_values) + 1), js_values, s=13, color=teal, edgecolor="white", linewidth=0.35)
    js_mean = summary["primary_macro_metrics"]["jensen_shannon_distance"]
    js_ci = boot["jensen_shannon_distance"]
    ax.axhline(js_mean, color=dark, linewidth=1.1)
    ax.fill_between(
        [0.5, len(js_values) + 0.5],
        js_ci["bootstrap_ci_low"],
        js_ci["bootstrap_ci_high"],
        color=teal,
        alpha=0.16,
        linewidth=0,
    )
    ax.set(xlim=(0.5, len(js_values) + 0.5), ylim=(0.0, max(0.34, js_values.max() + 0.02)), xlabel="Recordings (sorted)", ylabel="Jensen–Shannon distance")
    ax.text(0.04, 0.78, f"mean {js_mean:.3f}\n95% CI {js_ci['bootstrap_ci_low']:.3f}–{js_ci['bootstrap_ci_high']:.3f}", transform=ax.transAxes, color=dark)
    ax.set_title("Direct profile distance", fontsize=8, pad=4)
    ax.text(-0.18, 1.06, "b", transform=ax.transAxes, fontweight="bold", fontsize=9)

    ax = axes[2]
    labels = ["Dominant\npitch class", "Exact template\nlabel", "Tonic", "Mode"]
    proportions = np.array(
        [
            summary["primary_macro_metrics"]["dominant_pitch_class_match"],
            summary["primary_macro_metrics"]["descriptive_exact_label_agreement"],
            summary["primary_macro_metrics"]["descriptive_tonic_agreement"],
            summary["primary_macro_metrics"]["descriptive_mode_agreement"],
        ]
    )
    x = np.arange(len(labels))
    ax.bar(x, proportions, color=[blue, grey, grey, grey], width=0.68, edgecolor="none")
    for index, value in enumerate(proportions):
        ax.text(index, value + 0.025, f"{int(round(value * 38))}/38", ha="center", va="bottom", fontsize=6.7)
    ax.set(ylim=(0, 1.03), ylabel="Agreement proportion")
    ax.set_xticks(x, labels, fontsize=6.2)
    ax.axhline(0.5, color="#A8AFB6", linestyle=":", linewidth=0.7)
    ax.set_title("Agreement summaries", fontsize=8, pad=4)
    ax.text(0.02, 0.07, "Grey bars are descriptive\ntemplate-label agreement", transform=ax.transAxes, color="#59636D", fontsize=6.2)
    ax.text(-0.18, 1.06, "c", transform=ax.transAxes, fontweight="bold", fontsize=9)

    output_stem = OUTPUT_DIR / "fig_tonal_agreement"
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(MANUSCRIPT_DIR / "fig_tonal_agreement.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    qa = (
        "Core conclusion: note-derived and audio-derived pitch-class profiles agree closely, "
        "whereas template labels are descriptive rather than independent accuracy labels.\n"
        "Archetype: quantitative grid.\n"
        "Source data: per_recording_agreement.csv and summary.json.\n"
        "Statistics: n=38 recordings; means and 95% recording-bootstrap intervals use 10,000 resamples.\n"
        "Exports: SVG, PDF, TIFF (600 dpi), PNG.\n"
    )
    (OUTPUT_DIR / "qa_notes.txt").write_text(qa, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
