r"""
Initial pipeline: noisy depth image -> denoised -> restored (holes filled).

    noisy depth          depth image with noise and missing pixels (holes)
      |  0.4 m gate      depth closer than 0.4 m is treated as missing (a Kinect-type sensor cannot measure it)
      v
    denoising filter     median / invalid-aware median / Gaussian / bilateral (filters.py)
      |
      v
    denoised depth       noise reduced, holes may remain
      |  hole filling    FMM (Telea) / Navier-Stokes inpainting, or region fill (filters.py)
      v
    restored depth       every pixel (or every small hole) has a value

Usage
-----
    # any 16-bit depth PNG (value / 5000 = metres, 0 = missing)
    .venv\Scripts\python.exe denoise\pipeline.py --input path\to\depth.png

    # a dataset frame, with the stage figure
    .venv\Scripts\python.exe denoise\pipeline.py --icl 775 --figure
    .venv\Scripts\python.exe denoise\pipeline.py --tum 1200 --filter bilateral_cv2 --fill ns --figure

Options
-------
Default filter = the invalid-aware median (lowest error on ICL). Default fill = FMM, as in the project plan;
on ICL, Navier-Stokes ("--fill ns") gave more accurate fills (33 vs 130 mm, outputs/pipeline_icl/summary.csv).

    --filter    iamedian | median | gaussian_masked | gaussian_cv2 | bilateral_masked | bilateral_cv2 | none
    --k --sigma --d --sigma-space --sigma-range     filter settings (only the ones the filter uses)
    --fill      fmm | ns | region | none
    --radius    inpainting radius (px)        --max-hole-area   only fill holes up to this size (px)
    --gate      minimum depth in metres (0 = off)
    --out       output PNG (default: outputs/pipeline/<name>_restored.png)

In Python
---------
    from pipeline import restore
    stages = restore(depth_m)           # dict: input, gated, denoised, restored (float32 metres, NaN = missing)

Contents
--------
    restore         run the pipeline on one depth image, return every stage
    save_png        metres -> 16-bit PNG (x 5000, missing = 0)
    stage_figure    picture of every stage (with error maps when the true depth is known)
    main            command line
"""
import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D
import filters as F
from plots import INK2, SEQ, SURFACE


DEFAULT_FILTER = "iamedian"
DEFAULT_FILTER_PARAMS = {
    "iamedian": {"k": 9},
    "median": {"k": 11},
    "gaussian_masked": {"sigma_px": 1.5},
    "gaussian_cv2": {"sigma_px": 0.5},
    "bilateral_masked": {"d": 15, "sigma_space_px": 10, "sigma_range_m": 1.0},
    "bilateral_cv2": {"d": 15, "sigma_space_px": 10, "sigma_range_m": 0.3},
    "none": {},
}


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

def restore(depth, filter=DEFAULT_FILTER, filter_params=None, fill="fmm", fill_params=None, gate_m=0.4):
    """
    Run the initial pipeline on one depth image (float32 metres, NaN = missing).

    Returns {"input", "gated", "denoised", "restored"}: every stage, same format as the input.
    """
    if filter_params is None:
        filter_params = DEFAULT_FILTER_PARAMS[filter]
    fill_params = fill_params or {}

    gated = D.gate(depth, gate_m)
    denoised = gated if filter == "none" else F.FILTERS[filter](gated, **filter_params)

    prefixed = {f"fill_{k}": v for k, v in fill_params.items()}
    restored = F.pipeline(denoised, filter="none", fill=fill, **prefixed)

    return {"input": depth, "gated": gated, "denoised": denoised, "restored": restored}


def save_png(depth, path):
    """Metres -> 16-bit PNG in the dataset format (value / 5000 = metres, 0 = missing)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    u16 = np.round(np.nan_to_num(depth, nan=0.0) * D.DEPTH_SCALE)
    cv2.imwrite(path, np.clip(u16, 0, 65535).astype(np.uint16))
    print("wrote", path)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _missing_pct(d):
    return 100 * float((~D.valid(d)).mean())


def _image(ax, img, title, cmap=SEQ, vmin=0, vmax=None):
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, loc="left", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    return im


def stage_figure(stages, path, title, gt=None):
    """Row 1: input, denoised, restored depth. Row 2: error vs true depth (ICL) or what each step changed (TUM)."""
    inp, den, res = stages["input"], stages["denoised"], stages["restored"]
    vmax = np.nanpercentile(inp, 99)

    fig, axs = plt.subplots(2, 3, figsize=(15, 8.4), gridspec_kw={"hspace": 0.22, "wspace": 0.04})

    _image(axs[0, 0], inp, f"1. noisy input: {_missing_pct(inp):.1f} % missing (white)", vmax=vmax)
    _image(axs[0, 1], den, f"2. denoised: {_missing_pct(den):.1f} % missing", vmax=vmax)
    im = _image(axs[0, 2], res, f"3. restored (holes filled): {_missing_pct(res):.1f} % missing", vmax=vmax)
    fig.colorbar(im, ax=axs[0, :], fraction=0.012, pad=0.01, label="depth (m)")

    if gt is not None:
        # Error against the true depth, on the pixels each stage has
        for ax, d, name in zip(axs[1], (inp, den, res), ("input", "denoised", "restored")):
            err = np.abs(d - gt) * 1000
            im = _image(ax, err, f"|error| of {name}: mean {np.nanmean(err):.1f} mm", vmax=50)

        filled = D.valid(res) & ~D.valid(stages["gated"])
        if filled.any():
            fill_err = np.nanmean(np.abs(res - gt)[filled]) * 1000
            axs[1, 2].set_title(f"|error| of restored: mean {np.nanmean(np.abs(res - gt)) * 1000:.1f} mm "
                                f"(filled px: {fill_err:.0f} mm)", loc="left", fontsize=9)
        fig.colorbar(im, ax=axs[1, :], fraction=0.012, pad=0.01, label="|error| (mm, clipped at 50)")

    else:
        # No true depth: show what each step changed
        change = np.abs(den - inp) * 1000
        _image(axs[1, 0], change, f"change made by the filter: mean {np.nanmean(change):.1f} mm", vmax=50)

        filled = D.valid(res) & ~D.valid(den)
        _image(axs[1, 1], filled, f"pixels filled by hole filling (dark): {int(filled.sum()):,}", vmax=1)

        only_filled = np.where(filled, res, np.nan)
        _image(axs[1, 2], only_filled, "depth given to the filled pixels (check it looks plausible)", vmax=vmax)

    fig.suptitle(title, x=0.01, ha="left")
    fig.savefig(path, dpi=100, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Initial pipeline: noisy depth -> denoised -> restored")

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="16-bit depth PNG (value / 5000 = metres)")
    src.add_argument("--icl", type=int, help="ICL-NUIM noisy frame index (true depth used for the figure)")
    src.add_argument("--tum", type=int, help="TUM fr3 depth frame index")

    ap.add_argument("--filter", default=DEFAULT_FILTER, choices=list(DEFAULT_FILTER_PARAMS))
    ap.add_argument("--k", type=int)
    ap.add_argument("--sigma", type=float, help="Gaussian sigma (px)")
    ap.add_argument("--d", type=int, help="bilateral window diameter (px)")
    ap.add_argument("--sigma-space", type=float, help="bilateral sigma_space (px)")
    ap.add_argument("--sigma-range", type=float, help="bilateral sigma_range (m)")

    ap.add_argument("--fill", default="fmm", choices=list(F.FILLS))
    ap.add_argument("--radius", type=float, default=5, help="inpainting radius (px)")
    ap.add_argument("--max-hole-area", type=int, help="only fill holes up to this many pixels")

    ap.add_argument("--gate", type=float, default=0.4, help="minimum depth in metres, 0 = off")
    ap.add_argument("--out", help="output PNG")
    ap.add_argument("--figure", action="store_true", help="also save a picture of every stage")
    return ap.parse_args()


def main():
    a = parse_args()

    # Input
    gt = None
    if a.input:
        depth = D.png_to_m(cv2.imread(a.input, cv2.IMREAD_UNCHANGED))
        name = os.path.splitext(os.path.basename(a.input))[0]
    elif a.icl is not None:
        depth, gt = D.icl_noisy(a.icl), D.icl_gt(a.icl)
        name = f"icl_{a.icl:05d}"
    else:
        depth = D.tum_depth(a.tum)
        name = f"tum_{a.tum:05d}"

    # Settings: defaults for the chosen filter, overridden by any given on the command line
    filter_params = dict(DEFAULT_FILTER_PARAMS[a.filter])
    given = {"k": a.k, "sigma_px": a.sigma, "d": a.d, "sigma_space_px": a.sigma_space, "sigma_range_m": a.sigma_range}
    filter_params.update({k: v for k, v in given.items() if v is not None and k in filter_params})

    fill_params = {}
    if a.fill in ("fmm", "ns"):
        fill_params = {"radius_px": a.radius, "max_hole_area": a.max_hole_area}
    elif a.fill == "region" and a.max_hole_area:
        fill_params = {"max_hole_area": a.max_hole_area}

    # Run
    stages = restore(depth, a.filter, filter_params, a.fill, fill_params, a.gate)

    settings = ", ".join(f"{k} {v:g}" for k, v in filter_params.items())
    label = f"{a.filter} ({settings})" if settings else a.filter
    tag = f"{a.filter}_{a.fill}"

    out = a.out or os.path.join(D.OUTPUTS, "pipeline", f"{name}_{tag}_restored.png")
    save_png(stages["restored"], out)

    if a.figure:
        title = (f"Initial pipeline on {name}: gate {a.gate:g} m -> {label} -> {a.fill} hole filling")
        stage_figure(stages, os.path.splitext(out)[0].replace("_restored", "") + "_stages.png", title, gt)


if __name__ == "__main__":
    main()
