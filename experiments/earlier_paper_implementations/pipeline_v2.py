r"""
Pipeline V2: the Kinect-like depth denoising scheme (hole prediction + spatial-temporal DNBL).

    frames t-R .. t+R   depth video, float32 metres, NaN = missing
      |  0.4 m gate     closer than 0.4 m counts as missing
      v
    1. hole padding     temporary spatial estimates (Navier-Stokes) so that blocks around holes can be compared
      v
    2. block matching   for every pixel, the best-matching block in each neighbouring frame (and elsewhere in the
                        current frame), found by mean squared error (MSE) over the pixels measured in both
      v
    3. hole prediction  each hole pixel = weighted average of the matched reference values,
                        weight = exp(-MSE / 2 sigma^2); holes with no reliable match stay empty
      v
    4. ST-DNBL          bilateral filter over space (frame t) and time (matched blocks in neighbouring frames) whose
                        range weight is divided by the expected depth noise Theta(z) = noise_c * z^2
      v
    restored depth

Differences from V1 (pipeline.py): V1 denoises first and fills holes from one frame (spatial inpainting);
V2 fills holes first, from neighbouring frames, and then filters over space and time with a noise-aware range weight.

Usage
-----
    .venv\Scripts\python.exe research\pipeline_v2.py --icl 775 --figure
    .venv\Scripts\python.exe research\pipeline_v2.py --tum 1200 --figure --fallback ns

Options
-------
    --radius        neighbouring frames used on each side (default 2)
    --search        block-matching search radius in px (default 4)
    --block         matching block size in px (default 9)
    --sigma-match   sigma of the matching weight, metres (default 0.03)
    --noise-c       depth noise Theta(z) = noise_c * z^2 (default 0.0016: about 4 cm at 5 m)
    --kappa         range scale in units of Theta (default 3)
    --center        self | median: reference depth the range weight compares against
    --fallback      none | ns: what to do with holes no neighbouring frame can fill (default none, as in the paper)

In Python
---------
    from pipeline_v2 import restore_v2
    stages = restore_v2(frames, index)    # frames: list of depth images, index: the one to restore

Contents
--------
    block_match      best displacement and MSE of every pixel's block in one reference frame
    predict_holes    fill holes from the matched reference values
    dnbl             spatial-temporal divisive-normalised bilateral filter
    restore_v2       the whole pipeline on one frame of a sequence
    stage_figure_v2  picture of every stage
    main             command line
"""
import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))   # data.py lives here

import data as D
import filters as F
from pipeline import save_png, _image
from plots import SEQ, SURFACE


DEFAULTS = {
    "radius": 2,            # frames on each side
    "search": 4,            # px
    "block": 9,             # px
    "pad_weight": 0.25,     # weight of padded (estimated) pixels in the MSE
    "min_overlap": 0.25,    # fraction of the block that must be measured in both frames
    "sigma_match": 0.03,    # m
    "max_mse": 0.01,        # m^2: worse matches are ignored
    "min_weight": 0.1,      # total weight needed to fill a hole
    "noise_c": 0.0016,      # Theta(z) = noise_c * z^2
    "kappa": 3.0,           # range scale in units of Theta
    "sigma_space_px": 3.0,
    "spatial_r": 4,
    "temporal_r": 1,
    "center": "median",
    "fallback": "none",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shift(a, dy, dx, fill=0):
    """out[y, x] = a[y + dy, x + dx]; outside the image = fill."""
    H, W = a.shape
    out = np.full_like(a, fill)

    y0, y1 = max(0, -dy), min(H, H - dy)
    x0, x1 = max(0, -dx), min(W, W - dx)

    if y0 < y1 and x0 < x1:
        out[y0:y1, x0:x1] = a[y0 + dy:y1 + dy, x0 + dx:x1 + dx]

    return out


def _gather(ref, ref_ok, dy, dx):
    """Read ref at (y + dy, x + dx) for per-pixel integer maps dy, dx. Returns (values, measured)."""
    H, W = ref.shape
    yy, xx = np.mgrid[0:H, 0:W]
    y2, x2 = yy + dy, xx + dx

    inside = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
    y2 = np.clip(y2, 0, H - 1)
    x2 = np.clip(x2, 0, W - 1)

    return ref[y2, x2], ref_ok[y2, x2] & inside


def _displacements(search, include_zero):
    """Displacements ordered by distance, so that ties go to the smaller motion."""
    ds = [(dy, dx) for dy in range(-search, search + 1) for dx in range(-search, search + 1)]
    ds.sort(key=lambda d: (d[0] ** 2 + d[1] ** 2, d))
    return ds if include_zero else [d for d in ds if d != (0, 0)]


# ---------------------------------------------------------------------------
# Steps 1-2: padding and block matching
# ---------------------------------------------------------------------------

def pad_holes(depth):
    """Step 1: temporary spatial estimate for every hole (Navier-Stokes). Returns (padded, was_hole)."""
    ok = D.valid(depth)
    padded = F.inpaint(depth, method="ns", radius_px=5)
    return np.nan_to_num(padded, nan=0.0).astype(np.float32), ~ok


def block_match(cur_pad, cur_ok, ref, p):
    """
    Step 2: for every pixel, the displacement (dy, dx) whose block in ref has the lowest MSE against the
    block around that pixel in the current frame.

    MSE_i = (1/N) sum_k (A_k - B_ik)^2 over the pixels measured in the reference block and either measured
    (weight 1) or padded (weight pad_weight) in the current block.

    Returns (mse, dy, dx), mse = inf where no block overlaps enough. ref is the reference depth (NaN = missing).
    """
    ref_ok = D.valid(ref)
    r0 = np.nan_to_num(ref, nan=0.0).astype(np.float32)
    ref_f = ref_ok.astype(np.float32)

    k = p["block"]
    w_cur = np.where(cur_ok, 1.0, p["pad_weight"]).astype(np.float32)
    min_w = p["min_overlap"] * k * k

    best = np.full(cur_pad.shape, np.inf, np.float32)
    best_dy = np.zeros(cur_pad.shape, np.int32)
    best_dx = np.zeros(cur_pad.shape, np.int32)

    for dy, dx in p["_displacements"]:
        r_s = _shift(r0, dy, dx)
        w = w_cur * _shift(ref_f, dy, dx)

        num = cv2.boxFilter((cur_pad - r_s) ** 2 * w, -1, (k, k), normalize=False, borderType=cv2.BORDER_CONSTANT)
        den = cv2.boxFilter(w, -1, (k, k), normalize=False, borderType=cv2.BORDER_CONSTANT)

        mse = np.where(den >= min_w, num / np.maximum(den, 1e-6), np.inf)
        better = mse < best

        best[better] = mse[better]
        best_dy[better] = dy
        best_dx[better] = dx

    return best, best_dy, best_dx


# ---------------------------------------------------------------------------
# Step 3: hole prediction
# ---------------------------------------------------------------------------

def predict_holes(cur, refs, matches, p):
    """
    Step 3: I_hat(x, y) = sum_i w_i * I_ref_i(x + dx_i, y + dy_i) / sum_i w_i, with w_i = exp(-MSE_i / 2 sigma^2).

    Only measured reference pixels count, and only for matches with MSE <= max_mse. A hole gets a value only if
    the total weight reaches min_weight; the rest stay empty (the paper does not guess where no frame sees it).

    Returns (filled, predicted_mask).
    """
    ok = D.valid(cur)
    num = np.zeros(cur.shape, np.float64)
    den = np.zeros(cur.shape, np.float64)

    for ref, (mse, dy, dx) in zip(refs, matches):
        vals, measured = _gather(np.nan_to_num(ref, nan=0.0), D.valid(ref), dy, dx)
        w = np.exp(-mse / (2 * p["sigma_match"] ** 2)) * measured * (mse <= p["max_mse"])
        w = np.nan_to_num(w)

        num += w * vals
        den += w

    predicted = ~ok & (den >= p["min_weight"])

    filled = cur.copy()
    filled[predicted] = (num[predicted] / den[predicted]).astype(np.float32)
    return filled, predicted


# ---------------------------------------------------------------------------
# Step 4: spatial-temporal DNBL
# ---------------------------------------------------------------------------

def dnbl(cur, refs, matches, p):
    """
    Step 4: spatial-temporal divisive-normalised bilateral filter.

        I_out(p) = sum_q w_s(p, q) * w_r(p, q) * w_t(q) * I(q) / sum_q w_s * w_r * w_t

        w_s = exp(-|p - q|^2 / 2 sigma_space^2)
        w_r = exp(-(I(p) - I(q))^2 / (2 kappa^2 (Theta(p)^2 + Theta(q)^2)))      Theta(z) = noise_c * z^2
        w_t = 1 for pixels of the current frame; exp(-MSE / 2 sigma_match^2) for matched pixels of another frame

    The depth difference is divided by the expected noise at that depth ("divisive normalisation"), so the same
    filter smooths harder where the sensor is noisier and keeps small real steps where it is accurate.

    q runs over a spatial window in the current frame and a small window around the matched position in each
    reference frame. center "median" compares against the local median of p instead of p itself, so that a single
    spike cannot attract its own neighbours. Holes stay holes.
    """
    ok = D.valid(cur)
    z = np.where(ok, cur, 0).astype(np.float32)

    ref_z = z
    if p["center"] == "median":
        ref_z = np.where(ok, np.nan_to_num(F.iamedian(cur, k=5), nan=0.0), 0).astype(np.float32)

    theta = lambda d: p["noise_c"] * d * d
    two_k2 = 2 * p["kappa"] ** 2
    theta_p2 = theta(ref_z) ** 2
    sigma2 = 2 * p["sigma_space_px"] ** 2

    num = np.zeros(z.shape, np.float64)
    den = np.zeros(z.shape, np.float64)

    def add(vals, measured, w_extra, off2):
        nonlocal num, den
        d2 = (vals - ref_z) ** 2
        w_r = np.exp(-d2 / (two_k2 * (theta_p2 + theta(vals) ** 2) + 1e-12))
        w = np.exp(-off2 / sigma2) * w_r * w_extra * measured
        num += w * vals
        den += w

    # Spatial window in the current frame
    r = p["spatial_r"]
    ok_f = ok.astype(np.float32)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            add(_shift(z, dy, dx), _shift(ok_f, dy, dx), 1.0, dx * dx + dy * dy)

    # Temporal windows: around the matched position in every reference frame
    rt = p["temporal_r"]
    for ref, (mse, dy, dx) in zip(refs, matches):
        r0 = np.nan_to_num(ref, nan=0.0).astype(np.float32)
        r_ok = D.valid(ref)
        w_t = np.nan_to_num(np.exp(-mse / (2 * p["sigma_match"] ** 2))) * (mse <= p["max_mse"])

        for oy in range(-rt, rt + 1):
            for ox in range(-rt, rt + 1):
                vals, measured = _gather(r0, r_ok, dy + oy, dx + ox)
                add(vals, measured.astype(np.float32), w_t, ox * ox + oy * oy)

    out = np.full(z.shape, np.nan, np.float32)
    out[ok] = (num[ok] / np.maximum(den[ok], 1e-12)).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

def restore_v2(frames, index, gate_m=0.4, **overrides):
    """
    Run V2 on frames[index], using the other frames of the list as temporal references.

    frames   list of depth images (float32 metres, NaN = missing), consecutive in time
    index    which one to restore

    Returns {"input", "gated", "padded", "filled", "predicted", "restored"}:
    "filled" = after hole prediction, "predicted" = mask of pixels it filled, "restored" = final output.
    """
    p = dict(DEFAULTS)
    p.update(overrides)
    p["_displacements"] = _displacements(p["search"], include_zero=True)
    p["_displacements_cur"] = _displacements(p["search"], include_zero=False)

    gated = [D.gate(f, gate_m) for f in frames]
    cur = gated[index]
    cur_ok = D.valid(cur)

    # 1. padding
    cur_pad, _ = pad_holes(cur)

    # 2. matching: neighbouring frames, plus the current frame itself (zero displacement excluded)
    lo, hi = max(0, index - p["radius"]), min(len(frames), index + p["radius"] + 1)
    refs = [gated[i] for i in range(lo, hi) if i != index] + [cur]
    matches = []
    for ref in refs:
        q = dict(p)
        if ref is cur:
            q["_displacements"] = p["_displacements_cur"]
        matches.append(block_match(cur_pad, cur_ok, ref, q))

    # 3. hole prediction
    filled, predicted = predict_holes(cur, refs, matches, p)

    # Optional spatial fallback for holes no frame could fill
    if p["fallback"] == "ns":
        filled = F.inpaint(filled, method="ns", radius_px=5)

    # 4. spatial-temporal DNBL. The reference frames feed it as measured (their own holes are not filled).
    temporal_refs = [r for r in refs if r is not cur]
    temporal_matches = [m for r, m in zip(refs, matches) if r is not cur]
    restored = dnbl(filled, temporal_refs, temporal_matches, p)

    return {"input": frames[index], "gated": cur, "padded": cur_pad, "filled": filled,
            "predicted": predicted, "restored": restored}


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _missing_pct(d):
    return 100 * float((~D.valid(d)).mean())


def stage_figure_v2(stages, path, title, gt=None):
    """Row 1: input, after hole prediction, restored. Row 2: error vs true depth (ICL) or what each step changed."""
    inp, fil, res = stages["input"], stages["filled"], stages["restored"]
    vmax = np.nanpercentile(inp, 99)

    fig, axs = plt.subplots(2, 3, figsize=(15, 8.4), gridspec_kw={"hspace": 0.22, "wspace": 0.04})

    _image(axs[0, 0], inp, f"1. noisy input: {_missing_pct(inp):.1f} % missing (white)", vmax=vmax)
    _image(axs[0, 1], fil, f"2. holes predicted from other frames: {_missing_pct(fil):.1f} % missing", vmax=vmax)
    im = _image(axs[0, 2], res, f"3. after ST-DNBL (restored): {_missing_pct(res):.1f} % missing", vmax=vmax)
    fig.colorbar(im, ax=axs[0, :], fraction=0.012, pad=0.01, label="depth (m)")

    if gt is not None:
        for ax, d, name in zip(axs[1], (inp, fil, res), ("input", "hole-predicted", "restored")):
            err = np.abs(d - gt) * 1000
            im = _image(ax, err, f"|error| of {name}: mean {np.nanmean(err):.1f} mm", vmax=50)
        fig.colorbar(im, ax=axs[1, :], fraction=0.012, pad=0.01, label="|error| (mm, clipped at 50)")
    else:
        change = np.abs(res - stages["gated"]) * 1000
        _image(axs[1, 0], change, f"change made by the filter: mean {np.nanmean(change):.1f} mm", vmax=50)
        _image(axs[1, 1], stages["predicted"], f"pixels filled from other frames (dark): {int(stages['predicted'].sum()):,}", vmax=1)
        _image(axs[1, 2], np.where(stages["predicted"], res, np.nan), "depth given to the predicted pixels", vmax=vmax)

    fig.suptitle(title, x=0.01, ha="left")
    fig.savefig(path, dpi=100, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Pipeline V2: hole prediction + spatial-temporal DNBL")

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--icl", type=int, help="ICL-NUIM noisy frame index")
    src.add_argument("--tum", type=int, help="TUM fr3 depth frame index")

    ap.add_argument("--radius", type=int, default=DEFAULTS["radius"])
    ap.add_argument("--search", type=int, default=DEFAULTS["search"])
    ap.add_argument("--block", type=int, default=DEFAULTS["block"])
    ap.add_argument("--sigma-match", type=float, default=DEFAULTS["sigma_match"])
    ap.add_argument("--noise-c", type=float, default=DEFAULTS["noise_c"])
    ap.add_argument("--kappa", type=float, default=DEFAULTS["kappa"])
    ap.add_argument("--center", choices=("self", "median"), default=DEFAULTS["center"])
    ap.add_argument("--fallback", choices=("none", "ns"), default=DEFAULTS["fallback"])
    ap.add_argument("--gate", type=float, default=0.4)
    ap.add_argument("--out", help="output PNG")
    ap.add_argument("--figure", action="store_true")
    a = ap.parse_args()

    if a.icl is not None:
        n, load, i, gt, name = D.ICL_N_FRAMES, D.icl_noisy, a.icl, D.icl_gt(a.icl), f"icl_{a.icl:05d}"
    else:
        n, load, i, gt, name = D.tum_n_frames(), D.tum_depth, a.tum, None, f"tum_{a.tum:05d}"

    lo, hi = max(0, i - a.radius), min(n, i + a.radius + 1)
    frames = [load(j) for j in range(lo, hi)]

    stages = restore_v2(frames, i - lo, gate_m=a.gate, radius=a.radius, search=a.search, block=a.block,
                        sigma_match=a.sigma_match, noise_c=a.noise_c, kappa=a.kappa, center=a.center,
                        fallback=a.fallback)

    out = a.out or os.path.join(D.OUTPUTS, "pipeline_v2", f"{name}_restored.png")
    save_png(stages["restored"], out)

    if a.figure:
        title = f"Pipeline V2 on {name}: hole prediction from {len(frames) - 1} neighbouring frames -> ST-DNBL"
        stage_figure_v2(stages, os.path.splitext(out)[0].replace("_restored", "") + "_stages.png", title, gt)


if __name__ == "__main__":
    main()
