r"""
Figures from the result CSVs. Writes PNGs to outputs/figures/ and never changes the CSVs.

Usage
-----
    .venv\Scripts\python.exe denoise\plots.py tuning     tuning curves, one figure per family
    .venv\Scripts\python.exe denoise\plots.py eval       eval_selected: per-frame change vs raw, regions
    .venv\Scripts\python.exe denoise\plots.py visuals    saved frames of eval_selected: depth, |error|, cross-sections
    .venv\Scripts\python.exe denoise\plots.py tum        tum_selected diagnostics

Every number shown in a figure is in the matching summary.csv / per_frame.csv.

Contents
--------
    style               colours and matplotlib defaults
    tuning_k            MAE vs kernel size / sigma (median, iamedian, gaussian)
    tuning_bilateral    MAE vs sigma_range, one row per window size d
    tuning_hole_fill    fills vs bad fills vs hole area and edge guard
    eval_figs           per-frame change vs raw; MAE per region
    visuals             depth / |error| images and a cross-section per saved frame
    tum_figs            TUM change diagnostics
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D
from evaluate import gt_masks


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

FIG = os.path.join(D.OUTPUTS, "figures")

# Series colours in a fixed order (colour-blind checked); each series also gets its own marker.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]

INK = "#0b0b0b"         # main text
INK2 = "#52514e"        # axes, labels, reference lines
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

# Single-hue blue ramp for depth and |error| images (light = small, dark = large)
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

GATE_LABEL = {0: "no gate", 0.4: "gate 0.4 m"}

REGIONS = [
    ("all_mae_mm", "MAE, all matched px (mm)"),
    ("flat_mae_mm", "MAE, flat (>5 px from GT edge) (mm)"),
    ("band2_mae_mm", "MAE, 2 px GT-edge band (mm)"),
    ("coverage_pct_mean", "coverage of GT-valid px (%)"),
]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "font.size": 9,
    "axes.titlesize": 10,
    "legend.frameon": False,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    path = os.path.join(FIG, name)

    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    print("wrote", path)


def read_summary(run):
    return pd.read_csv(os.path.join(D.OUTPUTS, run, "summary.csv"))


def raw_line(ax, summary, col):
    """Dashed horizontal line at the raw value of col, labelled "raw"."""
    v = summary.loc[summary.config == "raw", col]
    if not len(v):
        return

    ax.axhline(v.iloc[0], color=INK2, lw=1, ls="--")
    ax.annotate("raw", (1, v.iloc[0]), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", color=INK2, fontsize=8)


def log_ticks(ax, values, base=10):
    """Log x axis with a tick and plain label at exactly the tested values."""
    values = sorted(values)

    ax.set_xscale("log", base=base)
    ax.set_xticks(values)
    ax.set_xticklabels([f"{v:g}" for v in values])
    ax.minorticks_off()


def config_axis(ax, names):
    """Configs on the y axis, top to bottom in experiment order (not ranked)."""
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()


# ---------------------------------------------------------------------------
# Tuning figures
# ---------------------------------------------------------------------------

def tuning_k(fam, x="p_k", xlabel="kernel size k (px)"):
    """One parameter on x (k or sigma), gate / no gate as two lines, one panel per region."""
    s = read_summary(f"tune_{fam}")
    f = s[s.method == fam]

    fig, axs = plt.subplots(1, 4, figsize=(15, 3.4))

    for ax, (col, label) in zip(axs, REGIONS):
        for j, (gate, d) in enumerate(f.groupby("min_depth_m")):
            d = d.sort_values(x)
            ax.plot(d[x], d[col], color=SERIES[j], marker=MARKERS[j], label=GATE_LABEL[gate])

        raw_line(ax, s, col)
        ax.set_title(label, loc="left")
        ax.set_xlabel(xlabel)

        if x == "p_k":
            ax.set_xticks(sorted(f[x].unique()))
        else:
            log_ticks(ax, f[x].unique(), base=2)

    axs[0].legend()
    fig.suptitle(f"{fam}: tuning frames 0:1509:50 ({int(f.frames.iloc[0])} frames), matched mask of this sweep",
                 x=0.01, ha="left")
    save(fig, f"tuning_{fam}.png")


def tuning_bilateral(fam, ext=False):
    """Rows = window size d, columns = region, lines = sigma_space, x = sigma_range. One figure per gate."""
    run = f"tune_{fam}_ext" if ext else f"tune_{fam}"
    s = read_summary(run)
    f = s[s.method == fam]

    for gate, fg in f.groupby("min_depth_m"):
        sizes = sorted(fg.p_d.unique())
        fig, axs = plt.subplots(len(sizes), 3, figsize=(13, 3.0 * len(sizes)), sharex=True)

        for r, d in enumerate(sizes):
            for c, (col, label) in enumerate(REGIONS[:3]):
                ax = axs[r, c]

                for j, (ss, q) in enumerate(fg[fg.p_d == d].groupby("p_sigma_space_px")):
                    q = q.sort_values("p_sigma_range_m")
                    ax.plot(q.p_sigma_range_m, q[col], color=SERIES[j], marker=MARKERS[j],
                            label=f"sigma_space {ss:g} px")

                raw_line(ax, s, col)
                log_ticks(ax, fg.p_sigma_range_m.unique())
                ax.set_title(f"d = {d:g} px - {label}", loc="left")

                if r == len(sizes) - 1:
                    ax.set_xlabel("sigma_range (m, log scale)")

        axs[0, 0].legend()
        fig.suptitle(f"{run}, {GATE_LABEL[gate]}: tuning frames 0:1509:50, matched mask of this sweep",
                     x=0.01, ha="left")

        suffix = "_ext" if ext else ""
        save(fig, f"tuning_{fam}{suffix}_{'gate0.4' if gate else 'nogate'}.png")


def tuning_hole_fill():
    """Rows = gate, columns = fill statistics, lines = edge guard, x = max hole area."""
    s = read_summary("tune_hole_fill")
    f = s[s.method == "hole_fill"].copy()
    f["guard"] = f.p_edge_guard_m.fillna(-1)  # -1 = no guard, so it groups and sorts first

    cols = [
        ("filled_mae_mm", "MAE of filled px with GT (mm)"),
        ("bad_fills_gt5cm_total", "filled px > 5 cm off (all frames)"),
        ("filled_px_total", "filled px (all frames)"),
        ("all_mae_mm", "MAE, all matched px (mm)"),
    ]
    gates = sorted(f.min_depth_m.unique())

    fig, axs = plt.subplots(len(gates), 4, figsize=(15, 3.2 * len(gates)), squeeze=False)

    for r, gate in enumerate(gates):
        for c, (col, label) in enumerate(cols):
            ax = axs[r, c]

            for j, (guard, q) in enumerate(f[f.min_depth_m == gate].groupby("guard")):
                q = q.sort_values("p_max_hole_area")
                name = "no edge guard" if guard < 0 else f"edge guard {guard:g} m"
                ax.plot(q.p_max_hole_area, q[col], color=SERIES[j], marker=MARKERS[j], label=name)

            log_ticks(ax, f.p_max_hole_area.unique(), base=2)
            ax.set_title(f"{GATE_LABEL[gate]} - {label}", loc="left")

            if r == len(gates) - 1:
                ax.set_xlabel("max hole area (px)")

    axs[0, 0].legend()
    fig.suptitle("hole_fill (k 5): tuning frames 0:1509:50", x=0.01, ha="left")
    save(fig, "tuning_hole_fill.png")


# ---------------------------------------------------------------------------
# Evaluation figures
# ---------------------------------------------------------------------------

def eval_figs(run="eval_selected"):
    pf = pd.read_csv(os.path.join(D.OUTPUTS, run, "per_frame.csv"))
    sm = read_summary(run)
    by_config = sm.set_index("config")
    order = list(sm.config)  # experiment order (family order), not ranked
    raw = pf[pf.config == "raw"].set_index("frame").all_mae_mm

    # 1. Per-frame change in MAE vs raw
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(order) + 1.2))

    for y, c in enumerate(order):
        d = pf[pf.config == c].set_index("frame").all_mae_mm - raw
        jitter = np.random.default_rng(y).uniform(-0.18, 0.18, len(d))

        ax.plot([d.quantile(0.1), d.quantile(0.9)], [y, y], color=SERIES[0], lw=2, solid_capstyle="round")
        ax.scatter(d, y + jitter, s=8, color=SERIES[0], alpha=0.45, lw=0)
        ax.plot(d.median(), y, marker="|", color=INK, ms=12, mew=2)

    ax.axvline(0, color=INK2, lw=1)
    config_axis(ax, order)
    ax.set_xlabel("per-frame MAE change vs raw (mm), matched mask; < 0 = lower error than raw")
    ax.set_title(f"{run}: dots = frames, bar = 10-90 % of frames, tick = median", loc="left")
    save(fig, f"{run}_delta_vs_raw.png")

    # 2. MAE per region: pooled value + spread over frames
    fig, axs = plt.subplots(1, 3, figsize=(15, 0.32 * len(order) + 1.2), sharey=True)

    for ax, (col, label) in zip(axs, REGIONS[:3]):
        region = col.replace("_mae_mm", "")

        for y, c in enumerate(order):
            s = pf[pf.config == c]
            per_frame = 1000 * s[f"{region}_sum_abs"] / s[f"{region}_n"]

            ax.plot([per_frame.quantile(0.1), per_frame.quantile(0.9)], [y, y],
                    color=SERIES[0], lw=2, solid_capstyle="round")
            ax.plot(by_config.loc[c, col], y, "o", color=SERIES[0], ms=6, mec=SURFACE, mew=2)

        ax.axvline(by_config.loc["raw", col], color=INK2, lw=1, ls="--")
        ax.set_title(label, loc="left")
        ax.set_xlabel("mm")

    config_axis(axs[0], order)
    fig.suptitle(f"{run}: dot = pooled MAE over all frames, bar = 10-90 % of per-frame MAE, dashed = raw",
                 x=0.01, ha="left")
    save(fig, f"{run}_regions.png")


def family_picks(gate=0.4):
    """The tuned config of each family for one gate, from outputs/tuning_selection.csv."""
    sel = pd.read_csv(os.path.join(D.OUTPUTS, "tuning_selection.csv"))
    return list(sel[sel.gate_m == gate].config)


def _no_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def visuals(run="eval_selected", show=None):
    """
    For every saved frame:
      - a 2 x 4 image grid: GT depth, raw |error|, |error| of one tuned config per family
      - a cross-section through the row with the most GT edge pixels: full row + zoom on its edges
    """
    if show is None:
        show = family_picks(0.4)

    for folder in sorted(glob.glob(os.path.join(D.OUTPUTS, run, "frames", "*"))):
        i = int(os.path.basename(folder))
        gt = np.load(os.path.join(folder, "gt.npy"))
        raw = np.load(os.path.join(folder, "raw.npy"))
        outs = {c: np.load(os.path.join(folder, f"{c}.npy")) for c in show}

        # --- Image grid -----------------------------------------------------
        panels = [("raw", raw)] + list(outs.items())
        fig, axs = plt.subplots(2, 4, figsize=(15, 6.2), gridspec_kw={"wspace": 0.04, "hspace": 0.18})
        axs = axs.flat

        axs[0].imshow(gt, cmap=SEQ, vmin=0, vmax=np.nanpercentile(gt, 99))
        axs[0].set_title(f"GT depth, frame {i}", loc="left", fontsize=8)
        _no_ticks(axs[0])

        for ax, (c, o) in zip(axs[1:], panels):
            err = np.abs(o - gt) * 1000
            mae = np.nanmean(err)
            im = ax.imshow(err, cmap=SEQ, vmin=0, vmax=50)
            ax.set_title(f"{c}\n|error|, frame MAE {mae:.1f} mm", loc="left", fontsize=8)
            _no_ticks(ax)

        cbar = fig.colorbar(im, ax=list(axs), fraction=0.015, pad=0.01)
        cbar.set_label("|error| (mm, clipped at 50); missing = blank")
        fig.suptitle(f"{run}, frame {i}: GT and |error| of raw and one tuned config per family (gate 0.4 m)",
                     x=0.01, ha="left")
        save(fig, f"{run}_frame{i:05d}_grid.png")

        # --- Cross-section --------------------------------------------------
        edge_rows = gt_masks(gt)["band1"]
        y = int(np.argmax(edge_rows.sum(1)))
        x = np.arange(gt.shape[1])

        # Zoom: the 120 px stretch of this row with the most edge pixels
        counts = np.convolve(edge_rows[y].astype(int), np.ones(120, int), mode="valid")
        x0 = int(np.argmax(counts))
        zoom = slice(x0, x0 + 120)

        n = len(outs)
        fig, axs = plt.subplots(n, 2, figsize=(14, 2.1 * n), sharey=True, sharex="col",
                                gridspec_kw={"width_ratios": [2.2, 1], "wspace": 0.05, "hspace": 0.35})

        for r, (c, o) in enumerate(outs.items()):
            for ax, xs in ((axs[r, 0], slice(None)), (axs[r, 1], zoom)):
                ax.plot(x[xs], raw[y, xs], ".", color=INK2, ms=2.5, label="raw")
                ax.plot(x[xs], gt[y, xs], color=INK, lw=2, label="GT")
                ax.plot(x[xs], o[y, xs], color=SERIES[0], lw=1.3, label=c)

            axs[r, 0].axvspan(x0, x0 + 120, color=GRID, alpha=0.6, lw=0)
            axs[r, 0].set_title(c, loc="left", fontsize=8)
            axs[r, 0].set_ylabel("depth (m)")

        axs[0, 0].legend(["raw (dots)", "GT", "filtered"], ncol=3, loc="upper left", fontsize=8)
        axs[0, 1].set_title(f"zoom: columns {x0}-{x0 + 119} (shaded on the left)", loc="left", fontsize=8)
        axs[-1, 0].set_xlabel("column (px)")
        axs[-1, 1].set_xlabel("column (px)")

        fig.suptitle(f"{run}, frame {i}, row {y} (the row with the most GT edge pixels); gaps = missing",
                     x=0.01, ha="left")
        save(fig, f"{run}_frame{i:05d}_row{y}.png")


# ---------------------------------------------------------------------------
# TUM figure
# ---------------------------------------------------------------------------

def tum_figs(run="tum_selected"):
    sm = read_summary(run)
    by_config = sm.set_index("config")

    cols = [
        ("change_flat_mm", "mean |out - in|, away from input edges (mm)"),
        ("change_edges_mm", "mean |out - in|, input-edge band (mm)"),
        ("flat_roughness_mm", "flat roughness (|Laplacian|/4) (mm)"),
        ("holes_pct", "missing px (%)"),
    ]

    fig, axs = plt.subplots(1, 4, figsize=(16, 0.32 * len(sm) + 1.2), sharey=True)

    for ax, (col, label) in zip(axs, cols):
        ax.plot(sm[col], range(len(sm)), "o", color=SERIES[0], ms=6, mec=SURFACE, mew=2)
        ax.axvline(by_config.loc["raw", col], color=INK2, lw=1, ls="--")
        ax.set_title(label, loc="left", fontsize=8)

    config_axis(axs[0], list(sm.config))
    fig.suptitle(f"{run}: TUM fr3 (no GT depth) - change relative to the input, NOT accuracy; dashed = raw",
                 x=0.01, ha="left")
    save(fig, f"{run}_diagnostics.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def all_tuning():
    for fam in ("median", "iamedian"):
        tuning_k(fam)

    for fam in ("gaussian_cv2", "gaussian_masked"):
        tuning_k(fam, x="p_sigma_px", xlabel="sigma (px); kernel 2*ceil(3 sigma)+1")

    tuning_hole_fill()

    for fam in ("bilateral_cv2", "bilateral_masked"):
        tuning_bilateral(fam)
        if os.path.exists(os.path.join(D.OUTPUTS, f"tune_{fam}_ext", "summary.csv")):
            tuning_bilateral(fam, ext=True)


COMMANDS = {"tuning": all_tuning, "eval": eval_figs, "visuals": visuals, "tum": tum_figs}


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "tuning"
    COMMANDS[what]()
