r"""
Checks on the ICL data and on the evaluation itself. Read-only on the data.

Usage
-----
    .venv\Scripts\python.exe denoise\audit.py [--frames 25:1509:50]

Checks (outputs/audit/)
-----------------------
    1  value conventions        zeros, values < 0.4 m, ranges, dtype and size (noisy and GT)
    2  alignment                MAE(raw, GT shifted by dx, dy) for |dx|, |dy| <= 2 px, flat and edge pixels
    3  error budget             share of the raw |error| from the GT-edge band, from < 0.4 m values, and the rest
    4  edge-pixel errors        same surface / wrong side / between surfaces / outside; and whether a
                                wrong value equals the GT of a pixel <= 1 px away
    5  bilateral check          OpenCV vs masked implementation; effect of holes entering OpenCV as 0 m

See audit2.py for the follow-up checks 6 and 7.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D
from evaluate import boundary_mixing, gt_masks
from filters import bilateral_cv2, bilateral_masked
from run import parse_frames


NEAR_M = 0.4                      # below this a Kinect cannot measure
BILATERAL_FRAMES = 10             # check 5 is slow; use the first 10 frames only
SIGMA_RANGES = (0.01, 0.03, 0.1, 0.3, 1.0)


def shifted(a, dy, dx):
    """out[y, x] = a[y + dy, x + dx]; pixels shifted in from outside are NaN."""
    out = np.full_like(a, np.nan)
    H, W = a.shape

    if dy >= 0:
        ys, yd = slice(dy, H), slice(0, H - dy)
    else:
        ys, yd = slice(0, H + dy), slice(-dy, H)

    if dx >= 0:
        xs, xd = slice(dx, W), slice(0, W - dx)
    else:
        xs, xd = slice(0, W + dx), slice(-dx, W)

    out[yd, xd] = a[ys, xs]
    return out


# ---------------------------------------------------------------------------
# The five checks, one frame at a time
# ---------------------------------------------------------------------------

def check_conventions(i, u_n, u_g, raw, gt, m):
    gv = D.valid(gt)
    rv = D.valid(raw)
    flat_ok = gv & rv & m["flat"] & (raw >= NEAR_M)

    return {
        "frame": i,
        "noisy_dtype": str(u_n.dtype),
        "gt_dtype": str(u_g.dtype),
        "shape_equal": u_n.shape == u_g.shape,
        "noisy_zero_px": int((u_n == 0).sum()),
        "gt_zero_px": int((u_g == 0).sum()),
        "noisy_below_0.4m_px": int(((u_n > 0) & (u_n < NEAR_M * D.DEPTH_SCALE)).sum()),
        "noisy_min_nonzero_m": float(u_n[u_n > 0].min()) / D.DEPTH_SCALE,
        "gt_min_m": float(u_g[u_g > 0].min()) / D.DEPTH_SCALE,
        "noisy_max_m": float(u_n.max()) / D.DEPTH_SCALE,
        "gt_max_m": float(u_g.max()) / D.DEPTH_SCALE,
        "noisy_values_multiple_of_5_(1mm)": bool((u_n[u_n > 0] % 5 == 0).all()),
        "median_signed_err_flat_mm": 1000 * float(np.nanmedian((raw - gt)[flat_ok])),
    }


def check_alignment(i, raw, gt, m):
    rv = D.valid(raw)
    rows = []

    for dy in range(-2, 3):
        for dx in range(-2, 3):
            g2 = shifted(gt, dy, dx)
            ok = rv & D.valid(g2) & (raw >= NEAR_M)
            e = np.abs(raw - g2)

            rows.append({
                "frame": i,
                "dy": dy,
                "dx": dx,
                "mae_flat_mm": 1000 * e[ok & m["flat"]].mean(),
                "mae_band2_mm": 1000 * e[ok & m["band2"]].mean(),
                "mae_all_mm": 1000 * e[ok].mean(),
            })

    return rows


def check_error_budget(i, raw, gt, m):
    M = D.valid(gt) & D.valid(raw)
    e = np.abs(raw - gt)

    near = M & (raw < NEAR_M)
    edge = M & m["band2"] & ~near
    rest = M & ~m["band2"] & ~near
    total = e[M].sum()

    return {
        "frame": i,
        "px": int(M.sum()),
        "mae_mm": 1000 * e[M].mean(),
        "share_near_lt0.4_pct": 100 * e[near].sum() / total,
        "share_band2_pct": 100 * e[edge].sum() / total,
        "share_rest_pct": 100 * e[rest].sum() / total,
        "px_near": int(near.sum()),
        "px_band2": int(edge.sum()),
        "px_rest": int(rest.sum()),
        "mae_rest_mm": 1000 * e[rest].mean(),
        "mae_band2_mm": 1000 * e[edge].mean(),
        "mae_excl_near_mm": 1000 * e[M & ~near].mean(),
    }


def check_edge_errors(i, raw, gt, m):
    M = D.valid(gt) & D.valid(raw)
    near = M & (raw < NEAR_M)
    e = np.abs(raw - gt)

    r = boundary_mixing(raw, gt, m["band2"], M & ~near)

    # Bad edge pixels: does the wrong value equal the GT of a pixel at most 1 px away?
    bad = M & ~near & m["band2"] & (e > 0.05)

    best = np.full(raw.shape, np.inf, np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            best = np.fmin(best, np.abs(raw - shifted(gt, dy, dx)))

    r.update({
        "frame": i,
        "band2_bad_px": int(bad.sum()),
        "bad_matching_gt_within_1px_pct": 100 * float((best[bad] <= 0.03).mean()) if bad.any() else np.nan,
        "bad_err_mean_mm": 1000 * float(e[bad].mean()) if bad.any() else np.nan,
    })
    return r


def check_bilateral(i, raw, gt):
    rv = D.valid(raw)
    gv = D.valid(gt)

    # "Interior": at least 9 px from any hole and 10 px from the image border
    interior = cv2.erode(rv.astype(np.uint8), np.ones((19, 19), np.uint8)) > 0
    interior[:10] = False
    interior[-10:] = False
    interior[:, :10] = False
    interior[:, -10:] = False

    near_holes = rv & ~interior
    rows = []

    for sr in SIGMA_RANGES:
        bc = bilateral_cv2(raw, 9, 5, sr)
        bm = bilateral_masked(raw, 9, 5, sr)
        both = D.valid(bc) & D.valid(bm)

        rows.append({
            "frame": i,
            "sigma_range_m": sr,
            "cv2_vs_masked_interior_maxdiff_mm": 1000 * float(np.abs(bc - bm)[interior & both].max()),
            "cv2_vs_masked_near_holes_or_border_meandiff_mm": 1000 * float(np.abs(bc - bm)[near_holes & both].mean()),
            "cv2_mae_near_holes_or_border_mm": 1000 * float(np.abs(bc - gt)[near_holes & both & gv].mean()),
            "masked_mae_near_holes_or_border_mm": 1000 * float(np.abs(bm - gt)[near_holes & both & gv].mean()),
        })

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="25:1509:50")
    args = ap.parse_args()

    out = os.path.join(D.OUTPUTS, "audit")
    os.makedirs(out, exist_ok=True)

    frames = parse_frames(args.frames, D.ICL_N_FRAMES)

    conventions = []
    alignment = []
    budget = []
    edge_rows = []
    bilateral = []

    for i in frames:
        u_n = cv2.imread(os.path.join(D.ICL_NOISY, f"{i}.png"), cv2.IMREAD_UNCHANGED)
        u_g = cv2.imread(os.path.join(D.ICL_CLEAN, f"{i}.png"), cv2.IMREAD_UNCHANGED)
        raw = D.png_to_m(u_n)
        gt = D.png_to_m(u_g)
        m = gt_masks(gt)

        conventions.append(check_conventions(i, u_n, u_g, raw, gt, m))
        alignment += check_alignment(i, raw, gt, m)
        budget.append(check_error_budget(i, raw, gt, m))
        edge_rows.append(check_edge_errors(i, raw, gt, m))

        if i in frames[:BILATERAL_FRAMES]:
            bilateral += check_bilateral(i, raw, gt)

        print(f"frame {i} done", flush=True)

    # Write
    pd.DataFrame(conventions).to_csv(os.path.join(out, "1_value_conventions.csv"), index=False)

    S = pd.DataFrame(alignment)
    S.to_csv(os.path.join(out, "2_alignment_shift_per_frame.csv"), index=False)
    S_mean = S.groupby(["dy", "dx"])[["mae_flat_mm", "mae_band2_mm", "mae_all_mm"]].mean().reset_index()
    S_mean.to_csv(os.path.join(out, "2_alignment_shift_mean.csv"), index=False)

    pd.DataFrame(budget).to_csv(os.path.join(out, "3_raw_error_budget.csv"), index=False)
    pd.DataFrame(edge_rows).to_csv(os.path.join(out, "4_raw_edge_errors.csv"), index=False)

    B = pd.DataFrame(bilateral)
    B.to_csv(os.path.join(out, "5_bilateral_cv2_vs_masked_per_frame.csv"), index=False)
    B_mean = B.groupby("sigma_range_m").mean().drop(columns="frame").reset_index()
    B_mean.to_csv(os.path.join(out, "5_bilateral_cv2_vs_masked.csv"), index=False)

    print("wrote", out)


if __name__ == "__main__":
    main()
