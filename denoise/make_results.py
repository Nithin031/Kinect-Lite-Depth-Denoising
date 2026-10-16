r"""
Build the RESULTS/ folder: the few pictures and one table worth looking at first.

Usage
-----
    .venv\Scripts\python.exe denoise\make_results.py

Run after the evaluation run, plots.py (visuals) and pointcloud.py (which writes the 3D viewers
straight into RESULTS/). Copies the chosen figures with
numbered names, draws the summary chart, and writes results_table.csv. RESULTS/README.md is
written by hand and is not touched here.
"""
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D
from plots import FIG, INK, INK2, SERIES, SURFACE

RESULTS = os.path.join(D.ROOT, "RESULTS")

# One tuned setting per family (0.4 m gate) + raw, in the order shown everywhere
SHOWN = [
    ("raw", "raw input (no filter)"),
    ("median_k11_gate0.4", "median, k 11"),
    ("iamedian_k9_gate0.4", "invalid-aware median, k 9"),
    ("gaussian_cv2_s0.5_gate0.4", "Gaussian, OpenCV, sigma 0.5 px"),
    ("gaussian_masked_s1.5_gate0.4", "Gaussian, masked, sigma 1.5 px"),
    ("bilateral_cv2_d15_s10_r0.3_gate0.4", "bilateral, OpenCV, sigma_r 0.3 m"),
    ("bilateral_masked_d15_s10_r1_gate0.4", "bilateral, masked, sigma_r 1 m"),
]

# Figure in outputs/ -> numbered name in RESULTS/
COPIES = [
    (os.path.join(FIG, "pc_1_depth_to_cloud.png"), "01_depth_image_to_point_cloud.png"),
    (os.path.join(FIG, "pc_2_noise_in_3d.png"), "02_what_the_noise_looks_like_in_3d.png"),
    (os.path.join(FIG, "theory_filters_1d.png"), "03_how_each_filter_works.png"),
    (os.path.join(FIG, "pc_3_filters_in_3d.png"), "05_filters_in_3d_icl.png"),
    (os.path.join(FIG, "eval_selected_frame00775_grid.png"), "06_error_images_icl_frame775.png"),
    (os.path.join(FIG, "pc_4_tum_real_sensor.png"), "07_real_sensor_tum.png"),
    (os.path.join(FIG, "pc_4b_tum_filters_in_3d.png"), "08_filters_in_3d_tum.png"),
    (os.path.join(D.OUTPUTS, "pipeline", "icl_00775_iamedian_fmm_stages.png"), "09_initial_pipeline_icl.png"),
    (os.path.join(D.OUTPUTS, "pipeline", "tum_01200_iamedian_fmm_stages.png"), "10_initial_pipeline_tum.png"),
]


def results_table():
    sm = pd.read_csv(os.path.join(D.OUTPUTS, "eval_selected", "summary.csv")).set_index("config")

    rows = []
    for config, name in SHOWN:
        r = sm.loc[config]
        rows.append({
            "filter": name,
            "error, all pixels (mm)": round(r.all_mae_mm, 1),
            "error, flat areas (mm)": round(r.flat_mae_mm, 1),
            "error, object edges (mm)": round(r.band2_mae_mm, 0),
            "change vs raw, median over frames (mm)": round(r.delta_vs_raw_median_mm, 1),
            "test frames worse than raw (of 30)": int(r.frames_worse_than_raw),
            "time per frame (ms)": round(r.runtime_ms_median, 0),
            "config id": config,
        })

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS, "results_table.csv"), index=False)
    return table


def summary_chart(table):
    """Flat-area and edge error side by side, medians in the accent colour."""
    names = list(table["filter"])
    is_median = ["median" in n for n in names]
    colours = [SERIES[0] if m else "#c3c2b7" for m in is_median]
    colours[0] = "#e4e3df"  # raw

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True, gridspec_kw={"wspace": 0.08})

    panels = [
        ("error, flat areas (mm)", "Flat areas (mm)", "{:.1f}"),
        ("error, object edges (mm)", "Object edges, 2 px band (mm)", "{:.0f}"),
    ]

    for ax, (col, title, fmt) in zip(axs, panels):
        values = table[col]
        y = range(len(values))

        ax.barh(y, values, color=colours, height=0.55)
        ax.axvline(values.iloc[0], color=INK2, lw=1, ls="--")

        for yi, v in zip(y, values):
            ax.text(v + values.max() * 0.015, yi, fmt.format(v), va="center", fontsize=9, color=INK2,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})

        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlim(0, values.max() * 1.15)
        ax.grid(axis="y", visible=False)

    axs[0].set_yticks(range(len(names)))
    axs[0].set_yticklabels(names)
    axs[0].invert_yaxis()

    fig.suptitle("Medians remove two-thirds of the edge error; averaging filters 4-30 %", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.text(0.01, 0.9, "Mean absolute error on 30 ICL test frames, one tuned setting per filter, 0.4 m gate. "
                        "Blue = median filters, dashed line = raw input.", fontsize=9, color=INK2)
    fig.subplots_adjust(top=0.8)

    path = os.path.join(RESULTS, "04_results_chart.png")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)


def main():
    os.makedirs(RESULTS, exist_ok=True)

    for src, name in COPIES:
        shutil.copyfile(src, os.path.join(RESULTS, name))
        print("copied", name)

    table = results_table()
    summary_chart(table)
    print(table.drop(columns="config id").to_string(index=False))


if __name__ == "__main__":
    main()
