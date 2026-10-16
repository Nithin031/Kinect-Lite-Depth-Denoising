"""
Evaluation masks and per-frame metrics.

ICL protocol (ground truth available)
-------------------------------------
- GT edge: a pixel whose noise-free depth jumps to a 4-neighbour by more than max(5 cm, 3 % of depth).
  Computed from the GT only, never from the noisy input or a filter output.
- Edge band b: GT edges dilated by b px (b = 1, 2, 3, 5). Band 2 is the original protocol.
- Flat: farther than 5 px from any GT edge. flat2: farther than 2 px (the original definition).
- Matched mask M: GT valid & input valid & output valid for EVERY config in the run.
  All accuracy numbers use M, so every config is scored on the same pixels.
- Own mask: GT valid & output valid for one config alone (shows the effect of filling / deleting pixels).

TUM protocol (no ground truth)
------------------------------
Only how much the output differs from the input. This is never an accuracy measure.

Contents
--------
    edges               GT depth-edge mask
    dilate              grow a mask by px pixels
    gt_masks            edge bands + flat masks for one GT frame
    boundary_mixing     classify wrong edge pixels (wrong side / between surfaces / outside)
    icl_frame_metrics   all ICL metrics for one frame and every config
    tum_frame_metrics   TUM change diagnostics for one frame and every config
"""
import cv2
import numpy as np

from data import valid


EDGE_ABS_M = 0.05   # a jump must exceed 5 cm ...
EDGE_REL = 0.03     # ... and 3 % of the nearer depth to count as an edge
BANDS = (1, 2, 3, 5)


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

def edges(depth, abs_m=EDGE_ABS_M, rel=EDGE_REL):
    """Mark both pixels of every right / down neighbour pair whose depth jump is an edge."""
    ok = valid(depth)
    H, W = depth.shape
    mask = np.zeros(depth.shape, bool)

    for dy, dx in ((0, 1), (1, 0)):
        a = depth[:H - dy, :W - dx]
        b = depth[dy:, dx:]
        both_ok = ok[:H - dy, :W - dx] & ok[dy:, dx:]

        with np.errstate(invalid="ignore"):
            threshold = np.maximum(abs_m, rel * np.minimum(a, b))
            jump = both_ok & (np.abs(a - b) > threshold)

        mask[:H - dy, :W - dx] |= jump
        mask[dy:, dx:] |= jump

    return mask


def dilate(mask, px):
    if px <= 0:
        return mask.copy()

    kernel = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def gt_masks(gt):
    """{"band1", "band2", "band3", "band5", "flat", "flat2"} for one GT frame."""
    e = edges(gt)

    out = {f"band{b}": dilate(e, b) for b in BANDS}
    out["flat"] = ~out["band5"]
    out["flat2"] = ~out["band2"]

    return out


# ---------------------------------------------------------------------------
# Error statistics
# ---------------------------------------------------------------------------

def _err(e):
    """Error statistics of a 1-D array of signed errors (metres in, mm out).

    n, sum_abs and sum_sq are kept so results can be pooled over frames exactly.
    """
    if e.size == 0:
        return {"n": 0, "sum_abs": 0.0, "sum_sq": 0.0}

    a = np.abs(e).astype(np.float64)

    return {
        "n": int(a.size),
        "sum_abs": float(a.sum()),
        "sum_sq": float((a * a).sum()),
        "mae_mm": 1000 * float(a.mean()),
        "rmse_mm": 1000 * float(np.sqrt((a * a).mean())),
        "medae_mm": 1000 * float(np.median(a)),
        "p95_mm": 1000 * float(np.percentile(a, 95)),
        "bias_mm": 1000 * float(e.mean()),
    }


def boundary_mixing(out, gt, band, M, tol=0.05, win=5):
    """
    What kind of value is a wrong edge pixel?

    The two surfaces at an edge are taken as the min and max GT depth in a win x win window.
    Only pixels where those differ by more than 2 * tol are used. A pixel is "bad" if it is
    more than tol from its own GT, and then:
      - wrong side: within tol of either surface (it took the other surface's depth)
      - between:    strictly between the two surfaces (a mixed / flying pixel)
      - outside:    neither (nearer or farther than both surfaces)
    Percentages are of all selected edge pixels.
    """
    g = np.where(valid(gt), gt, np.nan).astype(np.float32)
    kernel = np.ones((win, win), np.uint8)

    lo = -cv2.dilate(np.nan_to_num(-g, nan=-1e9), kernel)   # window minimum
    hi = cv2.dilate(np.nan_to_num(g, nan=-1e9), kernel)     # window maximum

    sel = band & M & (hi - lo > 2 * tol)
    o = out[sel]
    t = gt[sel]

    bad = np.abs(o - t) > tol
    near_lo = np.abs(o - lo[sel]) <= tol
    near_hi = np.abs(o - hi[sel]) <= tol

    wrong_side = bad & (near_lo | near_hi)
    between = bad & ~wrong_side & (o > lo[sel]) & (o < hi[sel])
    outside = bad & ~wrong_side & ~between

    n = max(1, int(sel.sum()))

    return {
        "edge_px_two_surface": int(sel.sum()),
        "edge_bad_pct": 100 * bad.sum() / n,
        "edge_wrong_side_pct": 100 * wrong_side.sum() / n,
        "edge_between_pct": 100 * between.sum() / n,
        "edge_outside_pct": 100 * outside.sum() / n,
    }


# ---------------------------------------------------------------------------
# Per-frame metrics
# ---------------------------------------------------------------------------

def icl_frame_metrics(gt, outs, raw_png_valid):
    """
    outs:           {config_id: output depth} for one frame
    raw_png_valid:  valid pixels of the ungated noisy PNG

    Returns {config_id: {metric: value}}.
    """
    gv = valid(gt)
    masks = gt_masks(gt)

    # Matched mask: valid in GT, in the input, and in every output.
    M = gv & raw_png_valid
    for o in outs.values():
        M &= valid(o)

    regions = [("all", M)]
    regions += [(f"band{b}", M & masks[f"band{b}"]) for b in BANDS]
    regions += [("flat", M & masks["flat"]), ("flat2", M & masks["flat2"])]

    res = {}
    for cid, o in outs.items():
        ov = valid(o)
        e = o - gt
        r = {"matched_px": int(M.sum())}

        # Accuracy on the matched mask, per region
        for name, sel in regions:
            for k, v in _err(e[sel]).items():
                r[f"{name}_{k}"] = v

        # Accuracy on this config's own valid pixels, and coverage
        own = gv & ov
        r["own_mae_mm"] = 1000 * float(np.abs(e[own]).mean()) if own.any() else np.nan
        r["coverage_pct"] = 100 * own.sum() / max(1, gv.sum())

        # Pixels the filter created (filled) or removed (lost)
        filled = ov & ~raw_png_valid
        filled_gt = filled & gv

        r["filled_px"] = int(filled.sum())
        r["filled_gt_px"] = int(filled_gt.sum())
        r["filled_mae_mm"] = 1000 * float(np.abs(e[filled_gt]).mean()) if filled_gt.any() else np.nan
        r["bad_fills_gt5cm"] = int((filled_gt & (np.abs(e) > 0.05)).sum())
        r["lost_px"] = int((raw_png_valid & ~ov).sum())

        # Kinds of wrong values in the 2 px edge band
        r.update(boundary_mixing(o, gt, masks["band2"], M))

        res[cid] = r

    return res


def tum_frame_metrics(inp, outs):
    """
    No GT: change of each output relative to the (ungated) input. Not accuracy.

    change_flat / change_edges  mean |output - input| away from / near input depth edges
    flat_roughness              mean |Laplacian| / 4 on flat valid pixels (lower = smoother)
    holes_pct, filled_px, lost_px
    """
    iv = valid(inp)
    band = dilate(edges(inp), 2)
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32) / 4

    res = {}
    for cid, o in outs.items():
        ov = valid(o)
        both = iv & ov
        change = np.abs(o - inp)

        z = np.nan_to_num(o, nan=0).astype(np.float32)
        lap = np.abs(cv2.filter2D(z, -1, laplacian, borderType=cv2.BORDER_REPLICATE))

        # Laplacian only where all 3x3 neighbours are valid (a missing neighbour would read as 0 m)
        lap_ok = cv2.erode(ov.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        flat = both & ~band & lap_ok

        res[cid] = {
            "change_flat_mm": 1000 * float(change[both & ~band].mean()),
            "change_edges_mm": 1000 * float(change[both & band].mean()) if (both & band).any() else np.nan,
            "flat_roughness_mm": 1000 * float(lap[flat].mean()) if flat.any() else np.nan,
            "holes_pct": 100 * float((~ov).mean()),
            "filled_px": int((ov & ~iv).sum()),
            "lost_px": int((iv & ~ov).sum()),
        }

    return res
