r"""
The paper pipeline: Fu et al., "Kinect-Like Depth Denoising", ISCAS 2012.

One implementation of the two earlier versions in experiments/earlier_paper_implementations/ (pipeline_v2.py and
pipeline_paper_kt.py). The only open question between them, how the hole's own frame (the i = 0 term of Eq. 7) takes part in the prediction, is the `center` switch.

    frames t-M .. t+M        float32 metres, NaN = hole
      |  0. gate             closer than 0.4 m = hole
      |  1. padding          small holes <- mean of the measured neighbours ("roughly padded", Fig. 4);
      |                      large voids (>= min_void px) are out-of-range holes and stay holes (Sec. III-A)
      |  2. block matching   per pixel and per neighbouring frame: the shift whose block has the minimum MSE
      |                      against the available (measured) pixels of the current block; searched blocks are padded
      |  3. hole prediction  Eq. (7)-(9): I_t(x, y) = sum_i w_i I_t+i(x + dx_i, y + dy_i) / sum_i w_i,
      |                      w_i = exp(-MSE_i^2 / 2 sigma_mse^2)
      |  4. ST-DNBL          Eq. (12), (14): bilateral filter over (2r + 1) motion-compensated k x k windows whose
      v                      range term is divided by the noise variance Theta(I(p)) = (noise_c * I(p)^2)^2
    restored frame t

The i = 0 term (center=...):
    paper     as written: i runs over [-m, m], so the current frame is searched too. There the block itself matches
              with MSE 0, so the hole gets its padded spatial mean with the largest weight of all terms.
    exclude   pipeline_paper_kt.py: skip i = 0, because the DNBL averages over space again. Holes that no other frame
              can predict keep the padded value.
    nonlocal  pipeline_v2.py: search the current frame but not at zero shift (non-local self-similarity); only
              measured centre values count. Holes nothing predicts keep the padded value.

Two readings that both implementations already share and are kept here:
    - Eq. (7) divides by 2m + 1, not by sum_i w_i; with w_i < 1 that would pull every hole toward 0, so the
      weighted average is normalised by sum_i w_i (the 1 / 2 pi sigma^2 of Eq. 9 then cancels).
    - Eq. (9) applies the Gaussian to the MSE itself, so the weight is exp(-MSE^2 / 2 sigma^2), sigma_mse in m^2.

Usage
-----
    .venv\Scripts\python.exe src\paper_pipeline.py --icl 775 --center paper
    .venv\Scripts\python.exe src\paper_pipeline.py --icl 775 --center all       (compares the three readings)

    from paper_pipeline import restore
    out = restore(frames, index, center="paper")      # dict of every stage, out["restored"] is the result
"""
import argparse
import os

import cv2
import numpy as np

import data as D

DEFAULTS = {             # k, sigma_r, sigma_mse tuned on ICL frames 600-1400 (every 100th), MAE 20.7 mm
    "gate": 0.4,          # m: closer = hole
    "m": 2,               # frames on each side for hole prediction (Eq. 7)
    "r": 1,               # frames on each side for the ST-DNBL (Eq. 14)
    "k": 9,               # block / window size (px)
    "search": 6,          # block-matching search radius (px)
    "min_void": 100,      # px: holes this large are out-of-range voids and are not filled
    "min_overlap": 0.25,  # fraction of a block that must be comparable for a match to count
    "sigma_mse": 1e-4,    # m^2: weight of a match, Eq. (8)-(9)
    "sigma_s": 2.0,       # px: spatial Gaussian of the DNBL
    "sigma_r": 16.0,      # range Gaussian in units of the noise std (16 * 2 mm = 3.2 cm at 1 m)
    "noise_c": 0.002,     # noise std at 1 m (m); Theta(z) = (noise_c * z^2)^2
    "center": "paper",    # paper | exclude | nonlocal: the i = 0 term of Eq. (7)
}
CENTERS = ("paper", "exclude", "nonlocal")


def has_value(depth):
    return np.isfinite(depth) & (depth > 0)


def gate(depth, min_m):
    out = depth.copy()
    out[depth < min_m] = np.nan
    return out


# ---------------------------------------------------------------------------
# 1. Padding
# ---------------------------------------------------------------------------

def pad(depth, k=5, min_void=100):
    """
    Fill small holes with the mean of the known pixels in the k x k window around them, growing inward until each
    small hole is closed. Returns (padded, void): void = large holes, left NaN.
    """
    ok = has_value(depth)
    _, labels, stats, _ = cv2.connectedComponentsWithStats((~ok).astype(np.uint8), connectivity=8)
    large = stats[:, cv2.CC_STAT_AREA] >= min_void
    large[0] = False                                   # label 0 = the measured pixels
    void = large[labels] & ~ok

    z = np.where(ok, depth, 0).astype(np.float32)
    known = ok.copy()
    todo = ~ok & ~void
    while todo.any():
        s = cv2.boxFilter(z, -1, (k, k), normalize=False, borderType=cv2.BORDER_CONSTANT)
        c = cv2.boxFilter(known.astype(np.float32), -1, (k, k), normalize=False, borderType=cv2.BORDER_CONSTANT)
        new = todo & (c > 0)
        if not new.any():
            break
        z[new] = s[new] / c[new]
        known |= new
        todo &= ~new

    return np.where(known, z, np.nan).astype(np.float32), void


# ---------------------------------------------------------------------------
# 2. Block matching
# ---------------------------------------------------------------------------

def _shifts(search, skip_zero=False):
    """All shifts in [-search, search]^2, nearest first, so that ties go to the smaller motion."""
    s = [(dy, dx) for dy in range(-search, search + 1) for dx in range(-search, search + 1)]
    s.sort(key=lambda d: (d[0] ** 2 + d[1] ** 2, d))
    return [d for d in s if d != (0, 0)] if skip_zero else s


def match(cur, ref_pad, p, skip_zero=False, ref_measured=None):
    """
    For every pixel of cur: the shift (dy, dx) of ref_pad whose k x k block has the minimum MSE over the pixels
    measured in cur (and known in ref_pad). Returns (mse, dy, dx, value, value_ok), value = ref_pad[y + dy, x + dx].
    mse = inf where fewer than min_overlap of the block can be compared. ref_measured (optional) restricts
    value_ok to measured reference pixels.
    """
    k, s = p["k"], p["search"]
    H, W = cur.shape
    a_ok = has_value(cur)
    a = np.where(a_ok, cur, 0).astype(np.float32)
    wa = a_ok.astype(np.float32)

    b_ok = has_value(ref_pad) if ref_measured is None else ref_measured
    bp = np.pad(np.nan_to_num(ref_pad), s, mode="edge")
    bk = np.pad(has_value(ref_pad).astype(np.float32), s, mode="edge")
    bo = np.pad(b_ok, s, mode="edge")
    min_n = p["min_overlap"] * k * k

    best = np.full((H, W), np.inf, np.float32)
    best_dy = np.zeros((H, W), np.int16)
    best_dx = np.zeros((H, W), np.int16)
    value = np.zeros((H, W), np.float32)
    value_ok = np.zeros((H, W), bool)

    for dy, dx in _shifts(s, skip_zero):
        b = bp[s + dy:s + dy + H, s + dx:s + dx + W]
        w = wa * bk[s + dy:s + dy + H, s + dx:s + dx + W]
        n = cv2.boxFilter(w, -1, (k, k), normalize=False, borderType=cv2.BORDER_CONSTANT)
        e = cv2.boxFilter((a - b) ** 2 * w, -1, (k, k), normalize=False, borderType=cv2.BORDER_CONSTANT)
        mse = np.where(n >= min_n, e / np.maximum(n, 1e-6), np.inf)

        better = mse < best
        best[better] = mse[better]
        best_dy[better] = dy
        best_dx[better] = dx
        value[better] = b[better]
        value_ok[better] = bo[s + dy:s + dy + H, s + dx:s + dx + W][better]

    return best, best_dy, best_dx, value, value_ok


# ---------------------------------------------------------------------------
# 3. Hole prediction
# ---------------------------------------------------------------------------

def predict_holes(cur, padded, void, matches, p, self_match=None):
    """
    Eq. (7)-(9) on the small holes of cur. matches = match() results of the neighbouring frames (i != 0).
    The i = 0 term depends on p["center"] (see the module docstring); self_match is the nonlocal search result.
    Returns (filled, predicted): predicted = holes given a value by the weighted average (not the padding fallback).
    """
    num = np.zeros(cur.shape, np.float64)
    den = np.zeros(cur.shape, np.float64)

    def add(mse, value, value_ok):
        w = np.exp(-(mse.astype(np.float64) ** 2) / (2 * p["sigma_mse"] ** 2)) * value_ok
        num[:] += w * value
        den[:] += w

    for mse, _, _, value, value_ok in matches:
        add(mse, value, value_ok)

    if p["center"] == "paper":                        # the block matches itself: MSE 0, value = padded
        add(np.zeros(cur.shape, np.float32), np.nan_to_num(padded), has_value(padded))
    elif p["center"] == "nonlocal":
        mse, _, _, value, value_ok = self_match
        add(mse, value, value_ok)

    hole = ~has_value(cur) & ~void
    predicted = hole & (den > 1e-12)
    fallback = hole & ~predicted                     # nothing matched: keep the rough spatial padding

    filled = cur.copy()
    filled[predicted] = (num[predicted] / den[predicted]).astype(np.float32)
    filled[fallback] = padded[fallback]
    return filled, predicted


# ---------------------------------------------------------------------------
# 4. Spatial-temporal DNBL
# ---------------------------------------------------------------------------

def dnbl(filled, refs, p):
    """
    Eq. (14): I(p) = 1/W_p sum_j sum_{q_j in S_t+j(p)} G_s(|p - q_j|) G'_r(|I(p) - I(q_j)|) I(q_j),
    with G'_r(d) = exp(-d^2 / (2 sigma_r^2 Theta(I(p)))) (Eq. 12), Theta(z) = (noise_c * z^2)^2.

    filled = frame t after hole prediction; refs = [(padded frame t+j, dy, dx)] for j != 0, (dy, dx) = per-pixel
    motion to that frame. S_t+j(p) is the k x k window around the motion-compensated position. Voids stay NaN.
    """
    H, W = filled.shape
    ok = has_value(filled)
    I_p = np.where(ok, filled, 0).astype(np.float64)
    theta = np.maximum((p["noise_c"] * I_p ** 2) ** 2, 1e-12)
    denom = 2 * p["sigma_r"] ** 2 * theta

    num = np.zeros((H, W), np.float64)
    den = np.zeros((H, W), np.float64)
    yy, xx = np.mgrid[0:H, 0:W]
    h = p["k"] // 2

    for img, dy, dx in [(filled, 0, 0)] + list(refs):
        z = np.nan_to_num(img)
        v = has_value(img)
        for wy in range(-h, h + 1):
            for wx in range(-h, h + 1):
                qy, qx = yy + dy + wy, xx + dx + wx
                inside = (qy >= 0) & (qy < H) & (qx >= 0) & (qx < W)
                qy, qx = np.clip(qy, 0, H - 1), np.clip(qx, 0, W - 1)
                I_q = z[qy, qx]
                w = (np.exp(-(wy * wy + wx * wx) / (2 * p["sigma_s"] ** 2))
                     * np.exp(-(I_p - I_q) ** 2 / denom) * (v[qy, qx] & inside))
                num += w * I_q
                den += w

    out = np.full((H, W), np.nan, np.float32)
    good = ok & (den > 0)
    out[good] = (num[good] / den[good]).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

def restore(frames, index, **overrides):
    """
    Restore frames[index] using its neighbours in the list (consecutive frames, float32 metres, NaN = hole).
    Returns {"gated", "padded", "void", "filled", "predicted", "restored"}.
    """
    p = dict(DEFAULTS)
    p.update(overrides)
    if p["center"] not in CENTERS:
        raise ValueError(f"center must be one of {CENTERS}")

    gated = [gate(f, p["gate"]) for f in frames]
    pads = [pad(g, p["k"], p["min_void"]) for g in gated]
    cur, (cur_pad, void) = gated[index], pads[index]

    # 2. block matching against every neighbour within m (their padded frames are searched)
    offsets = [i for i in range(-p["m"], p["m"] + 1) if i != 0 and 0 <= index + i < len(frames)]
    matches = {i: match(cur, pads[index + i][0], p) for i in offsets}
    self_match = None
    if p["center"] == "nonlocal":
        self_match = match(cur, cur_pad, p, skip_zero=True, ref_measured=has_value(cur))

    # 3. hole prediction
    filled, predicted = predict_holes(cur, cur_pad, void, list(matches.values()), p, self_match)

    # 4. ST-DNBL over frames t-r .. t+r, along the matched motion
    refs = [(pads[index + j][0], matches[j][1], matches[j][2]) for j in range(-p["r"], p["r"] + 1)
            if j != 0 and j in matches]
    restored = dnbl(filled, refs, p)

    return {"gated": cur, "padded": cur_pad, "void": void, "filled": filled, "predicted": predicted,
            "restored": restored}


def window(n_frames, index, p=None):
    """First and last+1 frame index restore() needs around `index`."""
    p = p or DEFAULTS
    half = max(p["m"], p["r"])
    return max(0, index - half), min(n_frames, index + half + 1)


# ---------------------------------------------------------------------------
# Command line: one ICL frame, error against the true depth
# ---------------------------------------------------------------------------

def _mm(a, gt, mask):
    m = mask & has_value(a) & has_value(gt)
    return 1000 * float(np.mean(np.abs(a[m] - gt[m]))) if m.any() else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Fu et al. 2012: hole prediction + spatial-temporal DNBL")
    ap.add_argument("--icl", type=int, required=True, help="ICL-NUIM noisy frame index")
    ap.add_argument("--center", default="paper", choices=CENTERS + ("all",))
    for name in ("m", "r", "k", "search"):
        ap.add_argument("--" + name, type=int, default=DEFAULTS[name])
    for name in ("sigma_mse", "sigma_s", "sigma_r", "noise_c"):
        ap.add_argument("--" + name.replace("_", "-"), dest=name, type=float, default=DEFAULTS[name])
    a = ap.parse_args()

    params = {n: getattr(a, n) for n in ("m", "r", "k", "search", "sigma_mse", "sigma_s", "sigma_r", "noise_c")}
    lo, hi = window(D.ICL_N_FRAMES, a.icl, params)
    frames = [D.icl_noisy(i) for i in range(lo, hi)]
    gt = D.icl_gt(a.icl)

    print(f"frame {a.icl}: input error {_mm(frames[a.icl - lo], gt, np.ones_like(gt, bool)):.2f} mm")
    for center in (CENTERS if a.center == "all" else (a.center,)):
        out = restore(frames, a.icl - lo, center=center, **params)
        holes = ~has_value(out["gated"])
        print(f"  center={center:8s}  restored {_mm(out['restored'], gt, np.ones_like(gt, bool)):6.2f} mm"
              f" | at holes: predicted {_mm(out['filled'], gt, holes):6.2f} mm, restored "
              f"{_mm(out['restored'], gt, holes):6.2f} mm | still missing "
              f"{100 * float((~has_value(out['restored'])).mean()):.2f} %")


if __name__ == "__main__":
    main()
