#!/usr/bin/env python3
"""Generate the complete revised manuscript figure bundle with Python.

All numeric panels read revision artifacts or explicitly archived exploratory
data.  The script writes editable SVG/PDF and 600-dpi PNG/TIFF exports together
with source-data files and a figure-level QA record.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as path_effects  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.io import wavfile  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revision" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jasm_revision.ground_truth import read_ground_truth  # noqa: E402
from jasm_revision.tonal import NOTE_NAMES  # noqa: E402


OUT = ROOT / "revision" / "results" / "manuscript_figures"
SOURCE = OUT / "source_data"
MANUSCRIPT = ROOT / "submission_asmp_latex_resubmission_source"
AUTHOR_FINAL = ROOT / "revision" / "assets" / "author_final"
FIG_DPI = 600

INK = "#24313A"
BLUE = "#2F6B9A"
BLUE_MID = "#6F9FBE"
BLUE_LIGHT = "#DCE9F1"
TEAL = "#4F9C96"
TEAL_LIGHT = "#DCEEEB"
ORANGE = "#D98245"
ORANGE_LIGHT = "#F4D8C3"
PURPLE = "#7C6A9E"
PURPLE_LIGHT = "#E8E1F0"
GREY = "#87939E"
GREY_LIGHT = "#EFF2F4"
BORDER = "#51616D"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "axes.linewidth": 0.75,
        "axes.edgecolor": BORDER,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def source_csv(name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(SOURCE / name, index=False)


def source_json(name: str, value: object) -> None:
    (SOURCE / name).write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def save_pub(fig: mpl.figure.Figure, name: str, *, copy_png: bool = True) -> None:
    """Save one figure in editable/vector and high-resolution raster formats."""

    stem = OUT / name
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=FIG_DPI,
        bbox_inches="tight",
        pil_kwargs={"compress_level": 3},
    )
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=FIG_DPI,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    if copy_png and MANUSCRIPT.is_dir():
        shutil.copyfile(stem.with_suffix(".png"), MANUSCRIPT / f"{name}.png")


def historical_frame_table() -> Path:
    config = yaml.safe_load(
        (ROOT / "revision" / "config" / "main_experiment.yaml").read_text(
            encoding="utf-8"
        )
    )
    return ROOT / config["data"]["historical_frame_table"]


def panel_label(ax: mpl.axes.Axes, label: str, x: float = -0.08, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=9,
        va="bottom",
        ha="left",
    )


def box(
    ax: mpl.axes.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    face: str = "white",
    edge: str = BORDER,
    lw: float = 1.0,
    fontsize: float = 7.0,
    radius: float = 0.025,
    weight: str = "normal",
    alpha: float = 1.0,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        transform=ax.transAxes,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        alpha=alpha,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.15,
    )


def arrow(
    ax: mpl.axes.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = BORDER,
    lw: float = 1.15,
    connection: str = "arc3",
    mutation: float = 10,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=mutation,
            linewidth=lw,
            color=color,
            connectionstyle=connection,
            shrinkA=2,
            shrinkB=2,
        )
    )


def fig01_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.75))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    line = "#111111"
    fill = "#FFFFFF"
    ax.text(0.5, 0.975, "Model architecture", fontsize=9, fontweight="bold", ha="center")
    ax.text(0.015, 0.90, "Note transcription branch", fontsize=8, fontweight="bold")
    ax.text(0.015, 0.33, "Independent tonal-representation branch", fontsize=8, fontweight="bold")

    box(ax, 0.015, 0.54, 0.115, 0.20, "Monophonic\nhumming\nwaveform", face=fill, edge=line)
    box(ax, 0.165, 0.54, 0.135, 0.20, "CREPE\nF0 + confidence\n10-ms frames", face=fill, edge=line)
    arrow(ax, (0.13, 0.63), (0.165, 0.63))

    box(ax, 0.335, 0.46, 0.17, 0.36, "", face=fill, edge=line, weight="bold")
    ax.text(
        0.42,
        0.735,
        "44-dimensional\nframe features",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.5,
        fontweight="bold",
    )
    ax.text(
        0.42,
        0.59,
        "pitch derivatives\nRMS energy\nspectral flux\nvoicing context\n±1/±3/±5 lags",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.5,
        color=line,
    )
    arrow(ax, (0.30, 0.63), (0.335, 0.63))

    box(ax, 0.545, 0.66, 0.14, 0.12, "RandomForest\nonset", face=fill, edge=line)
    box(ax, 0.545, 0.49, 0.14, 0.12, "RandomForest\noffset", face=fill, edge=line)
    arrow(ax, (0.505, 0.65), (0.545, 0.72), color=line)
    arrow(ax, (0.505, 0.60), (0.545, 0.55), color=line)
    box(ax, 0.72, 0.54, 0.13, 0.20, "Peak picking +\nonset–offset\npairing", face=fill, edge=line)
    arrow(ax, (0.685, 0.72), (0.72, 0.67), color=line)
    arrow(ax, (0.685, 0.55), (0.72, 0.61), color=line)
    box(ax, 0.88, 0.54, 0.105, 0.20, "Predicted notes:\nonset, offset,\nMIDI pitch", face=fill, edge=line, lw=1.2)
    arrow(ax, (0.85, 0.63), (0.88, 0.63))

    # Independent tonal branch: it shares the CREPE stream but has no feedback arrow.
    ax.plot([0.232, 0.232, 0.335], [0.54, 0.20, 0.20], transform=ax.transAxes, color=line, linewidth=1.0, linestyle="--")
    arrow(ax, (0.31, 0.20), (0.335, 0.20), color=line)
    box(ax, 0.335, 0.125, 0.19, 0.145, "Confidence-weighted\n12-bin pitch-class\nprofile", face=fill, edge=line)
    box(ax, 0.57, 0.125, 0.19, 0.145, "Direct profile agreement +\ndescriptive template label", face=fill, edge=line)
    arrow(ax, (0.525, 0.20), (0.57, 0.20), color=line)
    box(ax, 0.81, 0.125, 0.175, 0.145, "Cosine similarity,\nJensen–Shannon distance,\ntonic/mode", face=fill, edge=line)
    arrow(ax, (0.76, 0.20), (0.81, 0.20), color=line)
    ax.text(
        0.66,
        0.06,
        "No tonal feedback or hard pitch correction is applied to note transcription.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.3,
        color=line,
        style="italic",
    )
    fig.tight_layout(pad=0.25)
    save_pub(fig, "fig_pipeline")
    plt.close(fig)
    source_json(
        "fig01_pipeline.json",
        {
            "note_branch": [
                "waveform",
                "CREPE F0/confidence at 10-ms frames",
                "44-dimensional frame features",
                "separate onset/offset RandomForest classifiers",
                "peak picking and onset-offset pairing",
                "note onset, offset and MIDI pitch",
            ],
            "tonal_branch": [
                "confidence-weighted 12-bin pitch-class profile",
                "direct profile agreement and descriptive template labels",
            ],
            "independence_guardrail": "Tonal descriptors do not feed back into note transcription.",
        },
    )


def fig02_nested_cv() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.35))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    line = "#111111"
    fill = "#FFFFFF"
    ax.text(0.5, 0.965, "5-fold outer GroupKFold (group = recording)", fontsize=9, fontweight="bold", ha="center")

    box(ax, 0.02, 0.55, 0.11, 0.15, "38 annotated\nrecordings", face=fill, edge=line)
    box(ax, 0.16, 0.55, 0.12, 0.15, "Outer fold k:\nrecording-level\ntrain / test", face=fill, edge=line)
    arrow(ax, (0.15, 0.62), (0.18, 0.62))
    ax.text(0.22, 0.73, "partition once", transform=ax.transAxes, fontsize=6.2, ha="center")

    # The inner search is visibly contained in the outer-training region.
    ax.add_patch(
        FancyBboxPatch(
            (0.32, 0.52),
            0.40,
            0.33,
            transform=ax.transAxes,
            boxstyle="round,pad=0.01,rounding_size=0.018",
            facecolor=fill,
            edgecolor=line,
            linewidth=1.0,
            linestyle="--",
        )
    )
    ax.text(0.52, 0.815, "OUTER-TRAINING ONLY", transform=ax.transAxes, fontsize=7, fontweight="bold", ha="center")
    box(ax, 0.34, 0.67, 0.10, 0.105, "4-fold inner\nGroupKFold", face=fill, edge=line, fontsize=6.2)
    box(ax, 0.465, 0.67, 0.105, 0.105, "Inner OOF\nRF candidates\n(mean frame AP)", face=fill, edge=line, fontsize=6.2)
    box(ax, 0.595, 0.67, 0.105, 0.105, "Inner OOF\ndecoder grid\n(micro COnPOff)", face=fill, edge=line, fontsize=6.2)
    arrow(ax, (0.28, 0.63), (0.34, 0.72), color=line, connection="angle")
    arrow(ax, (0.44, 0.72), (0.465, 0.72), color=line)
    arrow(ax, (0.57, 0.72), (0.595, 0.72), color=line)
    box(ax, 0.405, 0.545, 0.235, 0.075, "Select RF + decoder using inner OOF only", face=fill, edge=line, fontsize=6.5, weight="bold")
    arrow(ax, (0.648, 0.67), (0.555, 0.62), color=line, connection="angle")

    box(ax, 0.76, 0.58, 0.20, 0.16, "Refit selected onset/offset RF\non all outer-training recordings\nwith frozen decoder parameters", face=fill, edge=line, fontsize=6.5)
    arrow(ax, (0.64, 0.58), (0.76, 0.66), color=line, connection="angle")

    # The locked test branch is outside the blue inner-search enclosure.
    box(ax, 0.35, 0.30, 0.18, 0.115, "LOCKED OUTER-TEST\nrecordings", face=fill, edge=line, lw=1.5, fontsize=7, weight="bold")
    arrow(ax, (0.22, 0.55), (0.35, 0.36), color=line, connection="angle")
    box(ax, 0.59, 0.30, 0.18, 0.115, "Predict outer-test\nprobabilities once\nwith frozen choices", face=fill, edge=line, fontsize=6.5)
    arrow(ax, (0.53, 0.36), (0.59, 0.36), color=line)
    # Refit connects to the prediction node only after the locked-test branch joins.
    ax.plot([0.86, 0.86, 0.68], [0.58, 0.47, 0.47], transform=ax.transAxes, color=line, linewidth=1.15)
    arrow(ax, (0.68, 0.47), (0.68, 0.415), color=line)
    box(ax, 0.82, 0.30, 0.14, 0.115, "Fold-k\nouter-test notes", face=fill, edge=line, fontsize=6.5, weight="bold")
    arrow(ax, (0.77, 0.36), (0.82, 0.36), color=line)

    box(ax, 0.30, 0.205, 0.40, 0.055, "No test labels or probabilities enter selection", face=fill, edge=line, fontsize=6.2)
    box(ax, 0.20, 0.105, 0.68, 0.065, "Repeat k = 1…5 → concatenate the five outer-test streams → final note-level metrics", face=fill, edge=line, fontsize=7, weight="bold")
    arrow(ax, (0.89, 0.30), (0.78, 0.17), color=line, connection="angle")
    ax.text(0.51, 0.045, "Outer-test recordings are each scored exactly once; model, threshold and decoder selection remain nested.", transform=ax.transAxes, fontsize=6.4, ha="center", style="italic", color=line)
    fig.tight_layout(pad=0.25)
    save_pub(fig, "fig_oof_protocol")
    plt.close(fig)
    source_json(
        "fig02_nested_cv.json",
        {
            "outer_splits": 5,
            "inner_splits": 4,
            "grouping_unit": "recording",
            "inner_model_selection": "mean frame average precision on inner OOF probabilities",
            "inner_decoder_selection": "micro COnPOff on inner OOF probabilities",
            "outer_test_use": "predict once after refitting on complete outer-training; never used for selection",
            "final_aggregation": "concatenate five outer-test prediction streams before scoring",
        },
    )


def load_method_comparison() -> pd.DataFrame:
    old = pd.read_csv(ROOT / "final_results_for_paper" / "method_comparison_clean.csv")
    old = old.rename(columns={"Methods": "method"})
    proposed = json.loads(
        (ROOT / "revision" / "results" / "molina_nested_cv" / "summary.json").read_text()
    )["micro"]
    rosvot = json.loads(
        (ROOT / "revision" / "results" / "rosvot_molina" / "summary.json").read_text()
    )["standard_molina_micro"]
    old = old.loc[~old["method"].eq("Ours-OnOff")].copy()
    rows = [
        {
            "method": "Proposed (nested CV)",
            "COnPOff": proposed["COnPOff_F"],
            "COnP": proposed["COnP_F"],
            "COn": proposed["COn_F"],
            "Split": proposed["Split"],
            "Merged": proposed["Merged"],
            "Spurious": proposed["Spurious"],
        },
        {
            "method": "ROSVOT (official checkpoint)",
            "COnPOff": rosvot["COnPOff_F"],
            "COnP": rosvot["COnP_F"],
            "COn": rosvot["COn_F"],
            "Split": rosvot["Split"],
            "Merged": rosvot["Merged"],
            "Spurious": rosvot["Spurious"],
        },
    ]
    # Keep the short, readable order used in the manuscript.
    published = old.loc[old["method"].str.startswith("Li et al.")].copy()
    heuristic = old.loc[old["method"].str.startswith("Heuristic")].copy()
    result = pd.concat([published, heuristic, pd.DataFrame(rows[::-1])], ignore_index=True)
    result["method_short"] = [
        "Li 1+2",
        "Li 1+3",
        "Li 1+2+3",
        "Heuristic",
        "ROSVOT",
        "Proposed",
    ]
    result["family"] = ["published", "published", "published", "heuristic", "external", "proposed"]
    return result


def fig03_method_comparison() -> None:
    data = load_method_comparison()
    source_csv("fig03_method_comparison.csv", data)
    colors = {
        "published": "#B7C9D5",
        "heuristic": GREY,
        "external": ORANGE,
        "proposed": BLUE,
    }
    metrics = [
        ("COnPOff", "COnPOff F-measure", "higher"),
        ("COnP", "COnP F-measure", "higher"),
        ("COn", "COn F-measure", "higher"),
        ("Split", "Split error", "lower"),
        ("Merged", "Merged error", "lower"),
        ("Spurious", "Spurious error", "lower"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.25), constrained_layout=True)
    for index, (metric, title, direction) in enumerate(metrics):
        ax = axes.flat[index]
        vals = data[metric].to_numpy(float)
        y = np.arange(len(data))
        ax.barh(y, vals, color=[colors[f] for f in data["family"]], height=0.62, edgecolor="white", linewidth=0.35)
        ax.set_yticks(y)
        if index % 3 == 0:
            ax.set_yticklabels(data["method_short"])
        else:
            ax.set_yticklabels([])
        ax.invert_yaxis()
        x_max = max(0.2 if metric in {"Split", "Merged", "Spurious"} else 0.8, float(vals.max()) * 1.24)
        ax.set_xlim(0, x_max)
        ax.set_title(title, fontsize=7.4, pad=3)
        ax.grid(axis="x", color="#D7DEE3", linewidth=0.45, alpha=0.75)
        ax.set_axisbelow(True)
        for yi, val in zip(y, vals):
            ax.text(val + x_max * 0.018, yi, f"{val:.3f}", va="center", fontsize=5.9)
        panel_label(ax, chr(ord("a") + index), x=-0.12 if index % 3 == 0 else -0.02, y=1.08)
        if index == 0:
            ax.text(0.0, 1.30, "Higher is better", transform=ax.transAxes, color=BLUE, fontsize=6.7, fontweight="bold")
        if index == 3:
            ax.text(0.0, 1.30, "Lower is better", transform=ax.transAxes, color=ORANGE, fontsize=6.7, fontweight="bold")
        if index % 3 == 0:
            ax.set_ylabel("Method")
        ax.set_xlabel("Score")
    fig.suptitle("Molina note-level transcription metrics", fontsize=8.5, x=0.52, y=1.015)
    save_pub(fig, "fig_method_comparison")
    plt.close(fig)


def fig04_ablation() -> None:
    path = ROOT / "revision" / "results" / "corrected_ablation" / "summary.csv"
    data = pd.read_csv(path)
    source_csv("fig04_corrected_ablation.csv", data)
    short = ["F0", "+ RMS", "+ flux", "+ context", "+ temporal lags"]
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(7.2, 2.75))
    ax.axvspan(3.5, 4.5, color=BLUE_LIGHT, alpha=0.55, zorder=0)
    series = [
        ("COnPOff_F_micro", "COnPOff", BLUE, "o", 0.018, "bottom"),
        ("COnP_F_micro", "COnP", TEAL, "s", -0.022, "top"),
        ("COn_F_micro", "COn", GREY, "^", 0.018, "bottom"),
    ]
    for col, label, color, marker, label_offset, vertical_alignment in series:
        vals = data[col].to_numpy(float)
        ax.plot(x, vals, color=color, marker=marker, markersize=4.5, linewidth=1.4, label=label)
        for xi, val in zip(x, vals):
            label_artist = ax.text(
                xi,
                val + label_offset,
                f"{val:.3f}",
                ha="center",
                va=vertical_alignment,
                fontsize=5.9,
                color=color,
                zorder=5,
            )
            label_artist.set_path_effects(
                [path_effects.withStroke(linewidth=1.8, foreground="white")]
            )
    vals = data["COnPOff_F_micro"].to_numpy(float)
    delta = vals[-1] - vals[-2]
    ax.annotate(
        f"largest COnPOff gain\n{delta:+.3f}",
        xy=(4, vals[-1]),
        xytext=(3.25, 0.43),
        fontsize=6.4,
        color=BLUE,
        ha="center",
        arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": 0.8},
    )
    ax.set_xticks(x, [f"{name}\n({int(n)} features)" for name, n in zip(short, data["n_features"])])
    ax.set_ylim(0.30, 0.96)
    ax.set_ylabel("Micro F-measure")
    ax.set_xlabel("Cumulative feature configuration")
    ax.grid(axis="y", color="#D7DEE3", linewidth=0.45)
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.02))
    ax.text(0.995, 0.02, "Exploratory; decoder operating point frozen", transform=ax.transAxes, ha="right", fontsize=6.2, color=GREY)
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.09, top=0.98, hspace=0.08)
    save_pub(fig, "fig_corrected_ablation")
    plt.close(fig)


def fig05_bootstrap() -> None:
    data = pd.read_csv(ROOT / "revision" / "results" / "molina_nested_cv" / "per_recording_metrics.csv")
    ci = pd.read_csv(ROOT / "revision" / "results" / "molina_secondary_analysis" / "recording_bootstrap_ci.csv")
    metrics = [("COnPOff_F", "COnPOff", BLUE), ("COnP_F", "COnP", TEAL), ("COn_F", "COn", GREY)]
    rng = np.random.default_rng(2026)
    rows = []
    samples_by_metric: dict[str, np.ndarray] = {}
    for metric, _, _ in metrics:
        values = data[metric].to_numpy(float)
        samples = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
        samples_by_metric[metric] = samples
        rows.extend({"metric": metric, "resample": i, "macro_mean": float(v)} for i, v in enumerate(samples))
    source_csv("fig05_bootstrap_samples.csv", pd.DataFrame(rows))
    source_csv("fig05_bootstrap_ci.csv", ci)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)
    for index, (metric, label, color) in enumerate(metrics):
        ax = axes[index]
        samples = samples_by_metric[metric]
        row = ci.loc[ci["metric"].eq(metric)].iloc[0]
        bins = np.linspace(samples.min() - 0.01, samples.max() + 0.01, 34)
        ax.hist(samples, bins=bins, color=color, alpha=0.72, edgecolor="white", linewidth=0.25, density=True)
        ax.axvspan(row["ci_low"], row["ci_high"], color=color, alpha=0.16, zorder=0)
        ax.axvline(row["estimate"], color=INK, linewidth=1.15)
        ax.axvline(row["ci_low"], color=INK, linestyle="--", linewidth=0.65)
        ax.axvline(row["ci_high"], color=INK, linestyle="--", linewidth=0.65)
        ax.set_title(label, fontsize=7.7, pad=3)
        ax.set_xlabel("Bootstrap macro F-measure")
        if index == 0:
            ax.set_ylabel("Density")
        ax.text(0.04, 0.92, f"mean {row['estimate']:.3f}\n95% CI {row['ci_low']:.3f}–{row['ci_high']:.3f}", transform=ax.transAxes, va="top", fontsize=6.1)
        panel_label(ax, chr(ord("a") + index), x=-0.12, y=1.07)
    fig.suptitle("Recording-level bootstrap (10,000 resamples; n = 38)", fontsize=8.5, y=1.055)
    save_pub(fig, "fig_bootstrap")
    plt.close(fig)


def load_child1_feature_data() -> pd.DataFrame:
    frame = pd.read_csv(historical_frame_table())
    frame.index.name = "row_index"
    probability = pd.read_csv(ROOT / "revision" / "results" / "molina_nested_cv" / "outer_test_frame_probabilities.csv.gz")
    probability = probability.set_index("row_index")
    probability = probability.rename(
        columns={
            "onset_label": "onset_label_corrected",
            "offset_label": "offset_label_corrected",
        }
    )
    columns = ["onset_probability", "offset_probability", "onset_label_corrected", "offset_label_corrected"]
    part = frame.loc[frame["stem"].eq("child1")].copy()
    part = part.join(probability[columns], how="left")
    return part.reset_index(names="row_index")


def fig06_features_child1() -> None:
    part = load_child1_feature_data()
    start, end = 0.55, 8.55
    view = part.loc[part["time"].between(start, end)].copy()
    sample_rate, waveform = wavfile.read(ROOT / "audio" / "child1.wav")
    waveform_dtype = waveform.dtype
    waveform = waveform.astype(float)
    if np.issubdtype(waveform_dtype, np.integer):
        waveform = waveform / np.iinfo(waveform_dtype).max
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    wav_time = np.arange(len(waveform)) / sample_rate
    wav_mask = (wav_time >= start) & (wav_time <= end)
    wav_time = wav_time[wav_mask]
    waveform = waveform[wav_mask]
    # Keep the source waveform and frame table traceable without bloating the plot CSV.
    source_csv("fig06_child1_frames.csv", view[["row_index", "time", "midi", "conf", "rms", "flux", "abs_dm", "local_pitch_std", "onset_probability", "offset_probability", "onset_label_corrected", "offset_label_corrected"]])
    stride = max(1, len(wav_time) // 5000)
    source_csv("fig06_child1_waveform.csv", pd.DataFrame({"time": wav_time[::stride], "amplitude": waveform[::stride]}))

    fig, axes = plt.subplots(4, 1, figsize=(7.2, 5.35), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.15, 0.95, 1.0], "hspace": 0.08})
    ax = axes[0]
    ax.plot(wav_time, waveform, color="#9BA3A8", linewidth=0.35, alpha=0.75, label="waveform")
    rms = view["rms"].to_numpy(float)
    rms_norm = (rms - rms.min()) / max(rms.max() - rms.min(), 1e-9)
    ax.plot(view["time"], rms_norm * 0.85, color=ORANGE, linewidth=1.0, label="RMS (scaled)")
    ax.set_ylabel("Amplitude")
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=6)
    panel_label(ax, "a", x=-0.06)

    ax = axes[1]
    conf = view["conf"].to_numpy(float)
    midi = view["midi"].to_numpy(float)
    valid = conf >= 0.30
    ax.plot(view["time"], midi, color="#B7C0C7", linewidth=0.45, alpha=0.8)
    ax.scatter(view.loc[valid, "time"], midi[valid], s=4, color=BLUE, alpha=0.78, linewidths=0)
    ax2 = ax.twinx()
    ax2.plot(view["time"], conf, color=TEAL, linewidth=0.85)
    ax2.axhline(0.30, color=TEAL, linestyle="--", linewidth=0.55)
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_ylabel("confidence", color=TEAL)
    ax.set_ylabel("MIDI pitch")
    ax.set_ylim(max(45, float(np.nanpercentile(midi, 1) - 2)), float(np.nanpercentile(midi, 99) + 2))
    panel_label(ax, "b", x=-0.06)

    ax = axes[2]
    abs_dm = view["abs_dm"].to_numpy(float)
    flux = view["flux"].to_numpy(float)
    abs_dm_norm = abs_dm / max(np.nanpercentile(abs_dm, 99), 1e-9)
    flux_norm = flux / max(np.nanpercentile(flux, 99), 1e-9)
    ax.plot(view["time"], np.clip(abs_dm_norm, 0, 1.2), color=PURPLE, linewidth=0.9, label="|Δ pitch|")
    ax.plot(view["time"], np.clip(flux_norm, 0, 1.2), color=BLUE_MID, linewidth=0.9, label="spectral flux")
    ax.set_ylim(-0.03, 1.23)
    ax.set_ylabel("scaled feature")
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=6)
    panel_label(ax, "c", x=-0.06)

    ax = axes[3]
    ax.plot(view["time"], view["onset_probability"], color=BLUE, linewidth=0.9, label="onset probability")
    ax.plot(view["time"], view["offset_probability"], color=ORANGE, linewidth=0.9, label="offset probability")
    ax.axhline(0.36, color=BLUE, linestyle="--", linewidth=0.5)
    ax.axhline(0.405, color=ORANGE, linestyle="--", linewidth=0.5)
    for time in view.loc[view["onset_label_corrected"].eq(1), "time"]:
        ax.axvline(time, color=BLUE, alpha=0.25, linewidth=0.55)
    for time in view.loc[view["offset_label_corrected"].eq(1), "time"]:
        ax.axvline(time, color=ORANGE, alpha=0.25, linewidth=0.55)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("boundary\nprobability")
    ax.set_xlabel("Time (s)")
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=6)
    panel_label(ax, "d", x=-0.06)
    axes[-1].set_xlim(start, end)
    axes[-1].set_xticks(np.arange(1, 9))
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.09, top=0.98, hspace=0.08)
    save_pub(fig, "fig_features_child1")
    plt.close(fig)


def clipped_notes(notes: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    notes = notes.loc[(notes["offset"] >= start) & (notes["onset"] <= end)].copy()
    notes["onset_plot"] = notes["onset"].clip(start, end)
    notes["offset_plot"] = notes["offset"].clip(start, end)
    return notes


def fig07_gt_prediction_child1() -> None:
    start, end = 0.55, 8.55
    reference = read_ground_truth(ROOT / "gt_files_temp" / "child1.GroundTruth.txt").notes
    prediction = pd.read_csv(ROOT / "revision" / "results" / "molina_nested_cv" / "outer_test_note_predictions.csv")
    prediction = prediction.loc[prediction["stem"].eq("child1")].copy()
    reference = clipped_notes(reference, start, end)
    prediction = clipped_notes(prediction, start, end)
    source_csv("fig07_child1_reference.csv", reference)
    source_csv("fig07_child1_prediction.csv", prediction)
    per_recording = pd.read_csv(ROOT / "revision" / "results" / "molina_nested_cv" / "per_recording_metrics.csv")
    score = per_recording.loc[per_recording["stem"].eq("child1")].iloc[0]

    y_all = np.concatenate([reference["midi"].to_numpy(float), prediction["midi"].to_numpy(float)])
    y_min = np.floor(np.nanmin(y_all) - 2)
    y_max = np.ceil(np.nanmax(y_all) + 2)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.15), sharex=True, sharey=True, gridspec_kw={"hspace": 0.12})
    for ax, notes, color, label in [(axes[0], reference, BLUE, "Reference notes"), (axes[1], prediction, ORANGE, "Outer-test prediction")]:
        for note in notes.itertuples(index=False):
            ax.add_patch(Rectangle((note.onset_plot, note.midi - 0.28), note.offset_plot - note.onset_plot, 0.56, facecolor=color, edgecolor=color, alpha=0.78, linewidth=0.4))
        ax.set_ylabel("MIDI pitch")
        ax.set_ylim(y_min, y_max)
        ax.grid(axis="y", color="#D7DEE3", linewidth=0.4)
        ax.text(0.01, 0.87, label, transform=ax.transAxes, fontsize=7, fontweight="bold", color=color)
    axes[0].text(0.99, 0.87, f"child1: COnPOff {score['COnPOff_F']:.3f}", transform=axes[0].transAxes, ha="right", fontsize=6.3, color=INK)
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(start, end)
    axes[-1].set_xticks(np.arange(1, 9))
    panel_label(axes[0], "a", x=-0.06)
    panel_label(axes[1], "b", x=-0.06)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.15, top=0.96, hspace=0.12)
    save_pub(fig, "fig_gt_prediction_child1")
    plt.close(fig)


def fig08_tonal_agreement() -> None:
    result_dir = ROOT / "revision" / "results" / "molina_tonal_agreement"
    data = pd.read_csv(result_dir / "per_recording_agreement.csv")
    summary = json.loads((result_dir / "summary.json").read_text())
    bootstrap = {item["metric"]: item for item in summary["bootstrap"]}
    source_csv("fig08_tonal_agreement.csv", data)
    source_csv("fig08_tonal_agreement_bootstrap.csv", pd.DataFrame(summary["bootstrap"]))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)
    values = np.sort(data["profile_cosine_similarity"].to_numpy(float))
    row = bootstrap["profile_cosine_similarity"]
    axes[0].scatter(np.arange(1, len(values) + 1), values, s=12, color=BLUE, edgecolor="white", linewidth=0.3)
    axes[0].axhline(row["macro_mean"], color=INK, linewidth=1.05)
    axes[0].axhspan(row["bootstrap_ci_low"], row["bootstrap_ci_high"], color=BLUE, alpha=0.14)
    axes[0].set(xlim=(0.5, len(values) + 0.5), ylim=(0.80, 1.005), xlabel="Recordings (sorted)", ylabel="Cosine similarity", title="Direct profile similarity")
    axes[0].text(0.04, 0.08, f"mean {row['macro_mean']:.3f}\n95% CI {row['bootstrap_ci_low']:.3f}–{row['bootstrap_ci_high']:.3f}", transform=axes[0].transAxes, fontsize=6.1)
    js = np.sort(data["jensen_shannon_distance"].to_numpy(float))
    row = bootstrap["jensen_shannon_distance"]
    axes[1].scatter(np.arange(1, len(js) + 1), js, s=12, color=TEAL, edgecolor="white", linewidth=0.3)
    axes[1].axhline(row["macro_mean"], color=INK, linewidth=1.05)
    axes[1].axhspan(row["bootstrap_ci_low"], row["bootstrap_ci_high"], color=TEAL, alpha=0.14)
    axes[1].set(xlim=(0.5, len(js) + 0.5), ylim=(0.0, max(0.34, float(js.max()) + 0.02)), xlabel="Recordings (sorted)", ylabel="Jensen–Shannon distance", title="Direct profile distance")
    axes[1].text(0.04, 0.78, f"mean {row['macro_mean']:.3f}\n95% CI {row['bootstrap_ci_low']:.3f}–{row['bootstrap_ci_high']:.3f}", transform=axes[1].transAxes, fontsize=6.1)
    labels = ["Dominant\npitch class", "Exact template\nlabel", "Tonic", "Mode"]
    proportions = np.array([summary["primary_macro_metrics"][key] for key in ("dominant_pitch_class_match", "descriptive_exact_label_agreement", "descriptive_tonic_agreement", "descriptive_mode_agreement")])
    x = np.arange(4)
    axes[2].bar(x, proportions, color=[BLUE, GREY, GREY, GREY], width=0.68)
    for i, value in enumerate(proportions):
        axes[2].text(i, value + 0.025, f"{int(round(value * 38))}/38", ha="center", fontsize=6.3)
    axes[2].set(ylim=(0, 1.03), ylabel="Agreement proportion", title="Agreement summaries")
    axes[2].set_xticks(x, labels, fontsize=5.6, rotation=32, ha="right")
    axes[2].axhline(0.5, color="#A8AFB6", linestyle=":", linewidth=0.7)
    axes[2].text(0.03, 0.98, "Grey bars: descriptive template labels", transform=axes[2].transAxes, fontsize=5.5, color=GREY, va="top")
    for i, ax in enumerate(axes):
        panel_label(ax, chr(ord("a") + i), x=-0.12, y=1.07)
    save_pub(fig, "fig_tonal_agreement")
    plt.close(fig)


def fig09_mode_distribution() -> None:
    result_dir = ROOT / "revision" / "results" / "all_audio_tonal_distribution"
    labels = pd.read_csv(result_dir / "per_recording_tonal_labels.csv")
    mode = pd.read_csv(result_dir / "mode_summary.csv")
    tonic_mode = pd.read_csv(result_dir / "tonic_mode_summary.csv")
    source_csv("fig09_tonal_labels.csv", labels)
    source_csv("fig09_mode_summary.csv", mode)
    source_csv("fig09_tonic_mode_summary.csv", tonic_mode)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55), gridspec_kw={"wspace": 0.33})
    order = ["major", "minor"]
    counts = [int(mode.loc[mode["mode"].eq(x), "count"].iloc[0]) for x in order]
    bars = axes[0].bar(order, counts, color=[BLUE, PURPLE], width=0.55)
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, count + 2, f"{count}\n({count / len(labels) * 100:.1f}%)", ha="center", va="bottom", fontsize=6.8)
    axes[0].set_ylim(0, max(counts) * 1.25)
    axes[0].set_ylabel("Recordings")
    axes[0].set_title("Descriptive mode labels", fontsize=7.8)
    pivot = tonic_mode.pivot(index="tonic", columns="mode", values="count").fillna(0).reindex(index=range(12), fill_value=0)
    x = np.arange(12)
    major = pivot.get("major", pd.Series(0, index=range(12))).to_numpy(float)
    minor = pivot.get("minor", pd.Series(0, index=range(12))).to_numpy(float)
    axes[1].bar(x, major, color=BLUE, width=0.62, label="major")
    axes[1].bar(x, minor, bottom=major, color=PURPLE, width=0.62, label="minor")
    axes[1].set_xticks(x, NOTE_NAMES, rotation=45, ha="right")
    axes[1].set_ylabel("Recordings")
    axes[1].set_title("Tonic distribution by mode", fontsize=7.8)
    axes[1].legend(frameon=False, ncol=2, fontsize=6, loc="upper right")
    panel_label(axes[0], "a", x=-0.12, y=1.05)
    panel_label(axes[1], "b", x=-0.12, y=1.05)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.90, wspace=0.33)
    save_pub(fig, "fig_mode_distribution")
    plt.close(fig)


def fig10_key_snapping_negative() -> None:
    data = pd.read_csv(ROOT / "final_results_for_paper" / "_archive_failed_experiments" / "postprocess_keysnap_ablation.csv")
    source_csv("fig10_historical_key_snapping.csv", data)
    labels = ["Baseline", "+ post-processing", "+ key-aware snap", "+ both"]
    values = data["COnPOff_F"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.2, 2.55))
    colors = [BLUE, ORANGE, ORANGE, ORANGE]
    bars = ax.bar(np.arange(4), values, color=colors, width=0.60, edgecolor="white", linewidth=0.4)
    for i, (bar, value) in enumerate(zip(bars, values)):
        delta = value - values[0]
        text = f"{value:.3f}" if i == 0 else f"{value:.3f}\nΔ {delta:+.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, text, ha="center", va="bottom", fontsize=6.4)
    ax.axhline(values[0], color=BLUE, linestyle="--", linewidth=0.8, alpha=0.8)
    ax.set_xticks(np.arange(4), labels)
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("COnPOff F-measure")
    ax.set_title("Historical exploratory result — original non-nested pipeline", fontsize=8.0, pad=25)
    ax.text(0.5, 1.035, "Design-provenance analysis; not part of revised confirmatory evaluation", transform=ax.transAxes, ha="center", va="bottom", fontsize=6.5, color=ORANGE, style="italic")
    fig.tight_layout(pad=0.45)
    save_pub(fig, "fig_key_snapping_negative")
    plt.close(fig)


def apply_author_schematic_overrides() -> None:
    """Install the author-approved raster redraws for the two schematics."""

    for name in ("fig_pipeline", "fig_oof_protocol"):
        source = AUTHOR_FINAL / f"{name}.png"
        if not source.is_file():
            continue

        target = OUT / f"{name}.png"
        shutil.copyfile(source, target)
        if MANUSCRIPT.is_dir():
            shutil.copyfile(source, MANUSCRIPT / f"{name}.png")

        with Image.open(source) as image:
            rgb = image.convert("RGB")
            rgb.save(
                OUT / f"{name}.tiff",
                compression="tiff_lzw",
                dpi=(FIG_DPI, FIG_DPI),
            )
            width, height = rgb.size
            export = plt.figure(
                figsize=(width / 300.0, height / 300.0),
                dpi=300,
                frameon=False,
            )
            axis = export.add_axes([0, 0, 1, 1])
            axis.imshow(rgb)
            axis.set_axis_off()
            export.savefig(OUT / f"{name}.pdf", dpi=300, pad_inches=0)
            export.savefig(OUT / f"{name}.svg", dpi=300, pad_inches=0)
            plt.close(export)


def write_qa_notes() -> None:
    text = """# Manuscript figure bundle QA

Backend: Python/matplotlib. Quantitative figures are exported as editable
SVG/PDF plus 600-dpi PNG/TIFF. Figures 1 and 2 use the author-approved final
raster redraws stored under `revision/assets/author_final/`; their matching
PDF/SVG/TIFF files embed the same raster artwork. Numeric panels read CSV/JSON
artifacts and the historical exploratory panel is explicitly marked as
non-nested provenance.

| Figure | Core conclusion | Archetype | Primary source data | Main reviewer-risk check |
|---|---|---|---|---|
| Fig. 1 | Note transcription and tonal representation are independent branches. | Schematic-led | fig01_pipeline.json | No tonal feedback arrow; tonal labels are descriptive. |
| Fig. 2 | Nested recording-grouped CV prevents outer-test selection leakage. | Schematic-led | fig02_nested_cv.json | Locked outer-test path is outside the inner-search container. |
| Fig. 3 | Proposed nested-CV result is shown alongside published Li et al., local heuristic and fixed ROSVOT reference values. | Quantitative grid | fig03_method_comparison.csv | Higher/lower metric directions and asymmetric ROSVOT training are explicit. |
| Fig. 4 | Temporal lags provide the largest cumulative ablation gain. | Quantitative trend | fig04_corrected_ablation.csv | Exploratory/frozen-decoder status is visible. |
| Fig. 5 | Recording-level macro estimates have reproducible bootstrap intervals. | Quantitative grid | fig05_bootstrap_samples.csv, fig05_bootstrap_ci.csv | n=38 and 10,000 resamples are explicit. |
| Fig. 6 | CREPE-derived features and boundary probabilities are traceable in a representative recording. | Image/quantitative composite | fig06_child1_frames.csv, fig06_child1_waveform.csv | Thresholds and boundary labels are shown. |
| Fig. 7 | Outer-test note intervals can be inspected against the audited reference. | Quantitative interval plot | fig07_child1_reference.csv, fig07_child1_prediction.csv | Reference and prediction use identical time/pitch axes. |
| Fig. 8 | Audio- and note-derived pitch-class profiles agree descriptively. | Quantitative grid | fig08_tonal_agreement.csv | Template labels are not called key accuracy. |
| Fig. 9 | Full-CREPE profiles yield 54 major and 78 minor descriptive labels over 132 recordings. | Quantitative grid | fig09_tonal_labels.csv, fig09_mode_summary.csv | Descriptive labels and lack of manual keys are explicit. |
| Fig. 10 | Hard tonal snapping degraded the historical pipeline. | Quantitative comparison | fig10_historical_key_snapping.csv | Historical non-nested status is prominent. |

Statistics and definitions are retained in the corresponding revision result
directories and manuscript captions.  No data under `final_results_for_paper/`
was overwritten.
"""
    (OUT / "qa_notes.md").write_text(text, encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    fig01_pipeline()
    fig02_nested_cv()
    fig03_method_comparison()
    fig04_ablation()
    fig05_bootstrap()
    fig06_features_child1()
    fig07_gt_prediction_child1()
    fig08_tonal_agreement()
    fig09_mode_distribution()
    fig10_key_snapping_negative()
    apply_author_schematic_overrides()
    write_qa_notes()
    print(f"Wrote figures to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
