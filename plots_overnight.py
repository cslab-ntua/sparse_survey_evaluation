#!/usr/bin/env python3
"""Plots of the overnight COO SpMM experiments (overnight.sh -> results.csv).

  1_speedup_vs_fusedmm   tier 1: speedup of every variant over FusedMM, per matrix, by dataset and k
  3_reordering_effect    k=256: speedup of each reordering over the same base kernel without reordering
  4_run_length           fixedk / row_split (same order) vs average run length of the order

Usage: python plots_overnight.py [results.csv]   (default: overnight_20261014_2306/results.csv)
Writes PNG + PDF figures and a CSV of the medians shown in each figure to <results dir>/plots/.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "overnight_20261014_2306/results.csv"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(RESULTS)), "plots")

# Reference palette (light): categorical slots in fixed order, ink / chrome tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # blue, orange, aqua (first three slots: valid all-pairs)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.facecolor": SURFACE,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

REORDER_NAMES = {
    "none": "none", "rcm": "RCM", "z_order": "z-order", "cross": "cross", "frontier": "frontier",
    "frontier_ringbuffer": "ring-buffer", "frontier_ringbuffer_all_seed": "ring-buffer\nall-seed",
    "frontier_ringbuffer_roulette": "ring-buffer\nroulette",
}
REORDER_ORDER = ["none", "RCM", "z-order", "cross", "frontier", "ring-buffer", "ring-buffer\nall-seed", "ring-buffer\nroulette"]
KS = [256, 512, 1024]


def split_variant(v):
    """Executable variant name -> (base, reordering)."""
    if v in ("mkl", "fusedmm"):
        return v, "none"
    v = v.replace("coo_vector_", "")
    if v == "xrow_row_split":
        return "row_split", "none"
    if v == "row_reg_fixedk":
        return "fixedk", "none"
    if v.startswith("fixedk_"):
        return "fixedk", REORDER_NAMES[v[len("fixedk_"):]]
    return "row_split", REORDER_NAMES[v]


def load():
    df = pd.read_csv(RESULTS, dtype={"sparsity": str})
    df = df[df["status"] == "ok"].copy()
    df["sparsity"] = df["sparsity"].fillna("")
    df[["base", "reorder"]] = df["variant"].apply(lambda v: pd.Series(split_variant(v)))
    df["key"] = df["dataset"] + "|" + df["matrix"] + "|" + df["sparsity"]
    return df


def log2_axis(ax, lo, hi):
    """Log y axis with ratio ticks (0.25, 0.5, 1, 2, ...)."""
    ax.set_yscale("log", base=2)
    ticks = [t for t in [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16] if lo / 1.5 <= t <= hi * 1.5]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}×" for t in ticks])
    ax.set_ylim(lo / 1.25, hi * 1.25)
    ax.minorticks_off()


def strip(ax, x, values, color, rng, width=0.28, size=10, alpha=0.55):
    """Jittered dots (with a surface ring) for the individual matrices."""
    jitter = rng.uniform(-width / 2, width / 2, len(values))
    ax.scatter(x + jitter, values, s=size, color=color, alpha=alpha, linewidths=0.4, edgecolors=SURFACE, zorder=2)


def median_tick(ax, x, value, width=0.36):
    ax.plot([x - width / 2, x + width / 2], [value, value], color=INK, linewidth=2, solid_capstyle="round", zorder=3)


def median_label(ax, x, value, fontsize=7.5):
    """Median value above its tick, on a surface-colored plate so it stays readable over the dots."""
    ax.annotate(f"{value:.2f}", (x, value), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom",
                fontsize=fontsize, color=INK, zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none", alpha=0.85))


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


#==========================================================================================================================================
# 1. Speedup over FusedMM (tier 1)
#==========================================================================================================================================

def plot_speedup_vs_fusedmm(df):
    t1 = df[df["tier"] == 1]
    fused = t1[t1["base"] == "fusedmm"].set_index(["key", "k"])["gflops"]
    t1 = t1[t1["base"] != "fusedmm"].copy()
    t1["speedup"] = t1["gflops"].values / fused.reindex(pd.MultiIndex.from_arrays([t1["key"], t1["k"]])).values
    t1 = t1.dropna(subset=["speedup"])

    variants = [("mkl", "none", "MKL"),
                ("row_split", "none", "row_split"), ("row_split", "z-order", "row_split\n+ z-order"), ("row_split", "RCM", "row_split\n+ RCM"),
                ("fixedk", "none", "fixedk"), ("fixedk", "z-order", "fixedk\n+ z-order"), ("fixedk", "RCM", "fixedk\n+ RCM")]
    family_color = {"row_split": SERIES[0], "fixedk": SERIES[1], "mkl": SERIES[2]}

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharey=True)
    rng = np.random.default_rng(0)
    medians = []
    for row, dataset in enumerate(["DLMC", "GRAPH"]):
        # Only the variants that ran on this dataset (RCM needs square matrices, so it has no DLMC results).
        present = set(zip(t1[t1["dataset"] == dataset]["base"], t1[t1["dataset"] == dataset]["reorder"]))
        ds_variants = [v for v in variants if (v[0], v[1]) in present]
        for col, k in enumerate(KS):
            ax = axes[row, col]
            sub = t1[(t1["dataset"] == dataset) & (t1["k"] == k)]
            labels = []
            for i, (base, reorder, label) in enumerate(ds_variants):
                vals = sub[(sub["base"] == base) & (sub["reorder"] == reorder)]["speedup"].values
                labels.append(label)
                if len(vals) == 0:
                    continue
                strip(ax, i, vals, family_color[base], rng, size=8 if dataset == "DLMC" else 18)
                med = float(np.median(vals))
                median_tick(ax, i, med)
                median_label(ax, i, med)
                medians.append({"dataset": dataset, "k": k, "variant": label.replace("\n", " "), "n": len(vals),
                                "median_speedup_vs_fusedmm": round(med, 3)})
            ax.axhline(1, color=INK_2, linewidth=1, zorder=1)
            ax.set_xticks(range(len(ds_variants)))
            ax.set_xticklabels(labels, fontsize=7.5)
            ax.set_xlim(-0.6, len(ds_variants) - 0.4)
            ax.grid(axis="x", visible=False)
            n_matrices = sub["key"].nunique()
            ax.set_title(f"{'DLMC' if dataset == 'DLMC' else 'Graphs'}, k = {k}  ({n_matrices} matrices)", loc="left")
            if col == 0:
                ax.set_ylabel("speedup over FusedMM (per matrix)")
    log2_axis(axes[0, 0], 1 / 8, 4)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=6, color=c, label=l)
               for l, c in [("row_split base", SERIES[0]), ("fixedk base", SERIES[1]), ("MKL", SERIES[2])]]
    handles.append(Line2D([], [], color=INK, linewidth=2, label="median (value above)"))
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), fontsize=8.5)
    fig.suptitle("Speedup over FusedMM per matrix (1× line = FusedMM)", y=1.06, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "1_speedup_vs_fusedmm")
    pd.DataFrame(medians).to_csv(os.path.join(OUT_DIR, "1_speedup_vs_fusedmm_medians.csv"), index=False)


#==========================================================================================================================================
# 3. Effect of each reordering on each base kernel (k = 256, the k where all reorderings ran)
#==========================================================================================================================================

def plot_reordering_effect(df, k=256):
    sub = df[(df["k"] == k) & df["base"].isin(["row_split", "fixedk"])]
    none = sub[sub["reorder"] == "none"].set_index(["key", "base"])["gflops"]
    sub = sub[sub["reorder"] != "none"].copy()
    sub["speedup"] = sub["gflops"].values / none.reindex(pd.MultiIndex.from_arrays([sub["key"], sub["base"]])).values
    sub = sub.dropna(subset=["speedup"])

    reorders = [r for r in REORDER_ORDER if r != "none" and r in set(sub["reorder"])]
    dataset_color = {"DLMC": SERIES[0], "GRAPH": SERIES[1]}
    offsets = {"DLMC": -0.18, "GRAPH": 0.18}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    rng = np.random.default_rng(1)
    medians = []
    for ax, base in zip(axes, ["row_split", "fixedk"]):
        for i, reorder in enumerate(reorders):
            for dataset in ["DLMC", "GRAPH"]:
                vals = sub[(sub["base"] == base) & (sub["reorder"] == reorder) & (sub["dataset"] == dataset)]["speedup"].values
                if len(vals) == 0:
                    continue
                x = i + offsets[dataset]
                strip(ax, x, vals, dataset_color[dataset], rng, width=0.26, size=8 if dataset == "DLMC" else 18)
                med = float(np.median(vals))
                median_tick(ax, x, med, width=0.3)
                median_label(ax, x, med, fontsize=7)
                medians.append({"base": base, "reordering": reorder.replace("\n", " "), "dataset": dataset, "n": len(vals),
                                "median_speedup_vs_no_reordering": round(med, 3)})
        ax.axhline(1, color=INK_2, linewidth=1, zorder=1)
        ax.set_xticks(range(len(reorders)))
        ax.set_xticklabels(reorders, fontsize=8)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{base} base, k = {k}", loc="left")
    axes[0].set_ylabel(f"speedup over the same base\nwithout reordering (per matrix)")
    log2_axis(axes[0], 1 / 4, 4)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=6, color=dataset_color["DLMC"], label="DLMC (259 matrices)"),
               Line2D([], [], marker="o", linestyle="", markersize=6, color=dataset_color["GRAPH"], label="Graphs (RCM: all 18, others: 9 up to ~2M nnz)"),
               Line2D([], [], color=INK, linewidth=2, label="median (value above)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03), fontsize=8.5)
    fig.suptitle("Effect of each reordering on each base kernel (above 1× = the reordering helps)", y=1.09, fontsize=11,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "3_reordering_effect")
    pd.DataFrame(medians).to_csv(os.path.join(OUT_DIR, "3_reordering_effect_medians.csv"), index=False)


#==========================================================================================================================================
# 4. fixedk vs row_split as a function of the average run length of the order
#==========================================================================================================================================

def plot_run_length(df):
    rs = df[df["base"] == "row_split"].set_index(["key", "k", "reorder"])["gflops"]
    fk = df[df["base"] == "fixedk"].copy()
    fk["ratio"] = fk["gflops"].values / rs.reindex(pd.MultiIndex.from_arrays([fk["key"], fk["k"], fk["reorder"]])).values
    fk = fk.dropna(subset=["ratio", "avg_run_length"])
    fk["order_class"] = np.where(fk["reorder"] == "none", "none (CSR order)",
                                 np.where(fk["reorder"] == "RCM", "RCM", "traversal (z-order, cross, frontier, ring-buffers)"))

    class_color = {"none (CSR order)": SERIES[0], "traversal (z-order, cross, frontier, ring-buffers)": SERIES[1], "RCM": SERIES[2]}
    dataset_marker = {"DLMC": "o", "GRAPH": "^"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    medians = []
    for ax, k in zip(axes, KS):
        sub = fk[fk["k"] == k]
        for cls, color in class_color.items():
            for dataset, marker in dataset_marker.items():
                pts = sub[(sub["order_class"] == cls) & (sub["dataset"] == dataset)]
                if len(pts) == 0:
                    continue
                ax.scatter(pts["avg_run_length"], pts["ratio"], s=14 if dataset == "DLMC" else 30, marker=marker, color=color,
                           alpha=0.6, linewidths=0.5, edgecolors=SURFACE, zorder=2)
                medians.append({"k": k, "order": cls, "dataset": dataset, "n": len(pts),
                                "median_avg_run_length": round(float(pts["avg_run_length"].median()), 2),
                                "median_fixedk_over_row_split": round(float(pts["ratio"].median()), 3)})
        ax.axhline(1, color=INK_2, linewidth=1, zorder=1)
        ax.set_xscale("log")
        ax.set_xlabel("average run length (non-zeros per run, log)")
        ax.set_title(f"k = {k}", loc="left")
        ax.text(0.98, 0.97, "fixedk faster", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=MUTED)
        ax.text(0.98, 0.03, "row_split faster", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED)
    axes[0].set_ylabel("fixedk / row_split GFLOPS\n(same matrix, same order)")
    log2_axis(axes[0], 1 / 4, 4)
    handles = [Line2D([], [], marker="s", linestyle="", markersize=7, color=c, label=l) for l, c in class_color.items()]
    handles += [Line2D([], [], marker=m, linestyle="", markersize=6, color=MUTED, label=d if d == "DLMC" else "Graphs")
                for d, m in dataset_marker.items()]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.04), fontsize=8.5)
    fig.suptitle("fixedk relative to row_split, against the average run length of the non-zero order", y=1.1, fontsize=11,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "4_run_length")
    pd.DataFrame(medians).to_csv(os.path.join(OUT_DIR, "4_run_length_medians.csv"), index=False)


if __name__ == "__main__":
    data = load()
    plot_speedup_vs_fusedmm(data)
    plot_reordering_effect(data)
    plot_run_length(data)
    print(f"plots written to {OUT_DIR}")
