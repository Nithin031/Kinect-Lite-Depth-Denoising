r"""
Follow-up checks on ICL (outputs/audit/6_*, 7_*). Read-only on the data.

Usage
-----
    .venv\Scripts\python.exe denoise\audit2.py [--frames 25:1509:50] [--workers 6]

Checks
------
6  "Outside" edge pixels: 2 px edge-band pixels of the raw input that are > 5 cm off and are
   neither near the min / max GT of the 5x5 window nor between them. For these:
   sign, size, whether a larger window (7 / 11 / 21 px) contains a matching GT surface,
   distance to the image border and to the nearest raw hole, and depth.

7  Sub-pixel alignment: MAE(raw, GT resampled at (y + dy, x + dx)) with bilinear interpolation,
   dy, dx in -2 .. +1 px in 0.25 px steps, on flat pixels only (> 5 px from a GT edge, so the
   interpolation never mixes two surfaces). Also per 3x3 image tile, to tell a constant offset
   from a scale / rotation difference.
"""
import argparse
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D
from evaluate import gt_masks
from run import parse_frames


TOL = 0.05                              # "bad" = more than 5 cm from GT
NEAR_M = 0.4
SHIFTS = np.arange(-2.0, 1.01, 0.25)    # sub-pixel shifts tested in check 7
WINDOWS = (5, 7, 11, 21)                # window sizes tested in check 6
MATCH_RADIUS = 10                       # px searched for a matching GT value in check 6


def window_min_max(g, win):
    """Min and max of g over a win x win window (NaN ignored)."""
    kernel = np.ones((win, win), np.uint8)

    lo = -cv2.dilate(np.nan_to_num(-g, nan=-1e9), kernel)
    hi = cv2.dilate(np.nan_to_num(g, nan=-1e9), kernel)
    return lo, hi


# ---------------------------------------------------------------------------
# Check 6: what are the "outside" edge pixels?
# ---------------------------------------------------------------------------

def outside_check(i, raw, gt, m):
    gv = D.valid(gt)
    rv = D.valid(raw)
    M = gv & rv & (raw >= NEAR_M)
    e = raw - gt

    g = np.where(gv, gt, np.nan).astype(np.float32)
    lo, hi = window_min_max(g, 5)

    # Same classification as evaluate.boundary_mixing
    sel = m["band2"] & M & (hi - lo > 2 * TOL)
    bad = sel & (np.abs(e) > TOL)
    near_surface = (np.abs(raw - lo) <= TOL) | (np.abs(raw - hi) <= TOL)
    between = bad & ~near_surface & (raw > lo) & (raw < hi)
    outside = bad & ~near_surface & ~between

    n = int(outside.sum())
    r = {
        "frame": i,
        "edge_px_two_surface": int(sel.sum()),
        "bad_px": int(bad.sum()),
        "outside_px": n,
        "outside_pct_of_bad": 100 * n / max(1, bad.sum()),
    }
    if not n:
        return r

    # Where are they?
    H, W = raw.shape
    yy, xx = np.mgrid[:H, :W]
    to_border = np.minimum(np.minimum(yy, H - 1 - yy), np.minimum(xx, W - 1 - xx))
    to_hole = cv2.distanceTransform(rv.astype(np.uint8), cv2.DIST_L2, 5)

    eo = e[outside]
    ov = raw[outside]
    go = gt[outside]

    r.update({
        "outside_below_window_min_pct": 100 * float((ov < lo[outside]).mean()),
        "outside_above_window_max_pct": 100 * float((ov > hi[outside]).mean()),
        "outside_err_mean_mm": 1000 * float(eo.mean()),
        "outside_abs_err_median_mm": 1000 * float(np.median(np.abs(eo))),
        "outside_abs_err_p90_mm": 1000 * float(np.percentile(np.abs(eo), 90)),
        "outside_gt_depth_median_m": float(np.median(go)),
        "outside_raw_depth_median_m": float(np.median(ov)),
        "outside_within_10px_of_border_pct": 100 * float((to_border[outside] < 10).mean()),
        "outside_within_2px_of_raw_hole_pct": 100 * float((to_hole[outside] <= 2).mean()),
        "outside_within_5px_of_raw_hole_pct": 100 * float((to_hole[outside] <= 5).mean()),
    })

    # Same distance for all selected edge pixels, as a baseline
    r["allsel_within_2px_of_raw_hole_pct"] = 100 * float((to_hole[sel] <= 2).mean())

    # Would a larger window contain a surface at that depth?
    for w in WINDOWS[1:]:
        lo_w, hi_w = window_min_max(g, w)
        inside = (ov >= lo_w[outside] - TOL) & (ov <= hi_w[outside] + TOL)
        r[f"outside_inside_range_win{w}_pct"] = 100 * float(inside.mean())

    # Does any GT value within MATCH_RADIUS px match the raw value to 3 cm?
    best = np.full(n, np.inf, np.float32)
    ys, xs = np.nonzero(outside)
    R = MATCH_RADIUS
    g_pad = np.pad(g, R, constant_values=np.nan)

    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            v = g_pad[ys + R + dy, xs + R + dx]
            best = np.fmin(best, np.abs(ov - v))

    r["outside_match_gt_within_10px_3cm_pct"] = 100 * float((best <= 0.03).mean())
    return r


# ---------------------------------------------------------------------------
# Check 7: sub-pixel alignment
# ---------------------------------------------------------------------------

def shift_check(i, raw, gt, m):
    H, W = raw.shape

    gt_valid = D.valid(gt).astype(np.float32)
    gt_zero = np.nan_to_num(gt, nan=0).astype(np.float32)
    ok0 = D.valid(raw) & (raw >= NEAR_M) & m["flat"]

    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    tile = (yy * 3 // H).astype(int) * 3 + (xx * 3 // W).astype(int)  # 0..8, row-major

    rows = []
    for dy in SHIFTS:
        for dx in SHIFTS:
            map_x = xx + np.float32(dx)
            map_y = yy + np.float32(dy)

            g2 = cv2.remap(gt_zero, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            v2 = cv2.remap(gt_valid, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            ok = ok0 & (v2 > 0.999)  # all four interpolation neighbours valid
            e = raw - g2

            r = {
                "frame": i,
                "dy": dy,
                "dx": dx,
                "n": int(ok.sum()),
                "mae_flat_mm": 1000 * float(np.abs(e[ok]).mean()),
                "medae_flat_mm": 1000 * float(np.median(np.abs(e[ok]))),
                "bias_flat_mm": 1000 * float(e[ok].mean()),
            }

            sums = np.bincount(tile[ok], np.abs(e[ok]), minlength=9)
            counts = np.bincount(tile[ok], minlength=9)
            for k in range(9):
                r[f"tile{k}_mae_mm"] = 1000 * sums[k] / counts[k] if counts[k] else np.nan

            rows.append(r)

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def work(i):
    raw = D.icl_noisy(i)
    gt = D.icl_gt(i)
    m = gt_masks(gt)
    return outside_check(i, raw, gt, m), shift_check(i, raw, gt, m)


def write_outside(O, out):
    O.to_csv(os.path.join(out, "6_outside_edge_pixels_per_frame.csv"), index=False)

    # Pool over frames, weighting each frame by its number of outside pixels
    w = O.outside_px
    pooled = {}
    for c in O.columns:
        if c.startswith("outside_") and c.endswith(("pct", "mm", "_m")) and c != "outside_pct_of_bad":
            pooled[c] = (O[c] * w).sum() / w.sum()

    pooled.update({
        "frames": len(O),
        "outside_px_total": int(w.sum()),
        "bad_px_total": int(O.bad_px.sum()),
        "outside_pct_of_bad": 100 * w.sum() / O.bad_px.sum(),
        "allsel_within_2px_of_raw_hole_pct": O.allsel_within_2px_of_raw_hole_pct.mean(),
    })

    s = pd.Series(pooled, name="value").rename_axis("metric (weighted by outside px per frame)")
    s.to_csv(os.path.join(out, "6_outside_edge_pixels.csv"))


def write_shift(S, out):
    S.to_csv(os.path.join(out, "7_subpixel_shift_per_frame.csv"), index=False)

    S_mean = S.drop(columns="frame").groupby(["dy", "dx"]).mean().reset_index()
    S_mean.to_csv(os.path.join(out, "7_subpixel_shift_mean.csv"), index=False)

    # Best shift per frame
    best = S.loc[S.groupby("frame").mae_flat_mm.idxmin(), ["frame", "dy", "dx", "mae_flat_mm"]]
    at_zero = S[(S.dy == 0) & (S.dx == 0)].set_index("frame").mae_flat_mm
    best["mae_flat_at_0_0_mm"] = best.frame.map(at_zero)
    best.to_csv(os.path.join(out, "7_subpixel_shift_best_per_frame.csv"), index=False)

    # Best shift per image tile (mean over frames)
    zero_row = S_mean[(S_mean.dy == 0) & (S_mean.dx == 0)]
    tiles = []
    for k in range(9):
        r = S_mean.loc[S_mean[f"tile{k}_mae_mm"].idxmin()]
        tiles.append({
            "tile (row*3+col)": k,
            "best_dy": r.dy,
            "best_dx": r.dx,
            "mae_at_best_mm": r[f"tile{k}_mae_mm"],
            "mae_at_0_0_mm": zero_row[f"tile{k}_mae_mm"].iloc[0],
        })
    pd.DataFrame(tiles).to_csv(os.path.join(out, "7_subpixel_shift_best_per_tile.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="25:1509:50")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    out = os.path.join(D.OUTPUTS, "audit")
    os.makedirs(out, exist_ok=True)

    for f in ("6_outside_edge_pixels.csv", "7_subpixel_shift_mean.csv"):
        if os.path.exists(os.path.join(out, f)):
            sys.exit(f"{f} exists; results are never overwritten")

    frames = parse_frames(args.frames, D.ICL_N_FRAMES)

    outside_rows = []
    shift_rows = []
    with Pool(args.workers) as pool:
        for k, (o, s) in enumerate(pool.imap_unordered(work, frames), 1):
            outside_rows.append(o)
            shift_rows += s
            print(f"{k}/{len(frames)}", flush=True)

    write_outside(pd.DataFrame(outside_rows).sort_values("frame"), out)
    write_shift(pd.DataFrame(shift_rows).sort_values(["frame", "dy", "dx"]), out)

    print("wrote", out)


if __name__ == "__main__":
    main()
