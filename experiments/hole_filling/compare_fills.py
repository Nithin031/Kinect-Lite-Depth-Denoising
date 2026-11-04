r"""
Hole-filling methods compared, after our gate + 9x9 median (src/clean.py).

    noisy depth -> gate -> 9x9 median -> fill method -> 1. depth error at the holes (ICL, true depth known)
                                                     -> 2. camera tracking, both trackers (ICL and TUM)

Fill methods (every one fills every hole; measured pixels never change):
    none          holes stay holes (our final choice so far)
    telea         OpenCV inpainting, Telea / fast marching (clean.fill_holes, the current "+ fill")
    navier-stokes OpenCV inpainting, Navier-Stokes (the earlier study's alternative)
    nearest       each hole pixel takes the nearest measured pixel's depth
    region (earlier) the earlier study's tuned region fill: small interior holes only, skipped at depth edges
    region median holes grow inward: a hole pixel with >= 3 measured / filled pixels in its 5x5 window takes
                  their median, repeated until the holes are closed (the earlier study's region fill)
    temporal (paper) the paper's block-matched hole prediction from frames t-2 .. t+2 with no spatial term
                  (the center=exclude reading, unpadded neighbours, no spatial fallback); unpredicted holes stay
    background    each hole takes the largest depth on its border. Kinect holes are mostly the shadow of a
                  foreground edge on the surface behind it, so the far side is the usual truth.
telea and navier-stokes are clamped to the depth range around their hole, as in clean.fill_holes.

Run from the project folder (datasets elsewhere: set KINECT_DATA to the folder holding "ICL NUIM" and "RGB TUM"):
    python experiments/hole_filling/compare_fills.py --jobs 10
Writes experiments/hole_filling/results/: fills.txt, fills.png, fill_example.png.
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import data as D                       # noqa: E402
import icl_compare as C                # noqa: E402
import tum_compare as T                # noqa: E402
from clean import fill_holes, gate, has_value, median_filter   # noqa: E402
from tum_poses import true_poses as tum_true_poses     # noqa: E402
import paper_pipeline as P                                       # noqa: E402

OUT = os.path.join(HERE, "results")


# ---------------------------------------------------------------------------
# Fill methods
# ---------------------------------------------------------------------------

def _holes(depth):
    ok = has_value(depth)
    n, hole_id = cv2.connectedComponents((~ok).astype(np.uint8), connectivity=8)
    ring = ndimage.grey_dilation(hole_id, size=(3, 3)) * ok        # measured pixels touching hole i get id i
    return ok, n, hole_id, ring


def inpaint_ns(depth, radius=5):
    ok, n, hole_id, ring = _holes(depth)
    if n == 1:
        return depth.copy()
    filled = cv2.inpaint(np.nan_to_num(depth).astype(np.float32), (~ok).astype(np.uint8), radius, cv2.INPAINT_NS)
    ids = np.arange(n)
    low = np.asarray(ndimage.minimum(depth, ring, ids), np.float32)
    high = np.asarray(ndimage.maximum(depth, ring, ids), np.float32)
    out = depth.copy()
    out[~ok] = np.clip(filled[~ok], low[hole_id][~ok], high[hole_id][~ok])
    return out


def nearest(depth):
    ok = has_value(depth)
    if ok.all():
        return depth.copy()
    _, (iy, ix) = ndimage.distance_transform_edt(~ok, return_indices=True)
    return depth[iy, ix]


def region_median(depth, k=5, min_neighbours=3, max_iters=400):
    out, filled = depth.copy(), has_value(depth)
    r = k // 2
    for _ in range(max_iters):
        todo = ~filled
        if not todo.any():
            break
        padded = np.pad(np.where(filled, out, np.nan), r, constant_values=np.nan)
        win = sliding_window_view(padded, (k, k))                  # view: no copy
        ys, xs = np.nonzero(todo & (ndimage.uniform_filter(filled.astype(np.float32), k) * k * k >= min_neighbours - 0.5))
        if len(ys) == 0:
            break
        vals = np.nanmedian(win[ys, xs].reshape(len(ys), -1), axis=1)
        good = np.isfinite(vals)
        if not good.any():
            break
        out[ys[good], xs[good]] = vals[good]
        filled[ys[good], xs[good]] = True
    return out


def region_earlier(depth, k=5, max_hole_area=1024, edge_guard_m=0.10, min_neighbours=3, max_iters=100):
    """
    The earlier study's region fill (research/filters.py hole_fill, tuned settings), copied unchanged in behaviour:
    only holes <= max_hole_area px that do not touch the image border; a pixel whose neighbours span more than
    edge_guard_m sits on a depth edge and is left empty. Fills few pixels, but only easy ones.
    """
    ok = has_value(depth)
    H, W = depth.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats((~ok).astype(np.uint8), connectivity=8)
    eligible = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        eligible[i] = area <= max_hole_area and not (x == 0 or y == 0 or x + w == W or y + h == H)
    target = eligible[labels] & ~ok
    out, filled, r = depth.copy(), ok.copy(), k // 2
    for _ in range(max_iters):
        todo = target & ~filled
        if not todo.any():
            break
        padded = np.pad(np.where(filled, out, np.nan), r, constant_values=np.nan)
        ys, xs = np.nonzero(todo)
        win = sliding_window_view(padded, (k, k))[ys, xs].reshape(len(ys), -1)
        n_valid = np.isfinite(win).sum(1)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(win, 1)
                spread = np.nanmax(win, 1) - np.nanmin(win, 1)
        cand = n_valid >= min_neighbours
        edge = cand & (spread > edge_guard_m)
        target[ys[edge], xs[edge]] = False                 # never retried
        cand &= ~edge
        if not cand.any():
            break
        out[ys[cand], xs[cand]] = med[cand]
        filled[ys[cand], xs[cand]] = True
    return out


def background(depth):
    ok, n, hole_id, ring = _holes(depth)
    if n == 1:
        return depth.copy()
    far = np.asarray(ndimage.maximum(depth, ring, np.arange(n)), np.float32)
    out = depth.copy()
    out[~ok] = far[hole_id][~ok]
    return out


def _load(dataset, i):
    """Frame i after gate + median, or None if it does not exist (sequence ends, ICL subset on Kaggle)."""
    try:
        if dataset == "icl":
            if not os.path.exists(os.path.join(D.ICL_NOISY, f"{i}.png")):
                return None
            return median_filter(gate(D.icl_noisy(i)))
        return median_filter(gate(D.tum_depth(i))) if 0 <= i < D.tum_n_frames() else None
    except (IOError, OSError):
        return None


def temporal_paper(depth, ctx):
    """
    The paper's hole prediction without any spatial averaging (a friend's suggestion; center=exclude reading):
    every hole of frame t is block-matched (k x k MSE, Eq. 7-9) in frames t-2 .. t+2 after the same median, and
    gets the MSE-weighted average of the matched values. The i = 0 term (the spatially padded value) is left out,
    neighbours are not padded, and holes that no neighbour predicts stay holes: no value comes from the
    surrounding pixels of the same frame.
    """
    dataset, i = ctx
    p = dict(P.DEFAULTS, center="exclude")
    matches = []
    for j in range(i - p["m"], i + p["m"] + 1):
        ref = None if j == i else _load(dataset, j)
        if ref is not None:
            matches.append(P.match(depth, ref, p))
    if not matches:
        return depth.copy()
    no_void = np.zeros(depth.shape, bool)
    filled, _ = P.predict_holes(depth, depth, no_void, matches, p)   # padded = depth: unpredicted holes stay NaN
    return filled


SPATIAL = {
    "none": lambda d: d,
    "telea (current)": fill_holes,
    "navier-stokes": inpaint_ns,
    "nearest": nearest,
    "region (earlier)": region_earlier,
    "region median": region_median,
    "background": background,
}
FILLS = {name: (lambda f: (lambda d, ctx: f(d)))(f) for name, f in SPATIAL.items()}
FILLS["temporal (paper)"] = temporal_paper          # needs the neighbouring frames: ctx = (dataset, frame)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def _icl_frame(i, keep=False):
    """Median-filtered frame i, every fill, and the hole errors against the true depth (mm)."""
    gt = D.icl_gt(i)
    base = median_filter(gate(D.icl_noisy(i)))
    holes = ~has_value(base) & has_value(gt)
    scores, outs = {}, {}
    for name, fill in FILLS.items():
        t0 = time.perf_counter()
        out = fill(base, ("icl", i))
        ms = 1000 * (time.perf_counter() - t0)
        e = np.abs(out - gt)[holes & has_value(out)] * 1000
        scores[name] = {"hole_mae": float(e.mean()) if e.size else np.nan,
                        "hole_median": float(np.median(e)) if e.size else np.nan,
                        "within_5cm": 100 * float(np.mean(e < 50)) if e.size else np.nan,
                        "filled": 100 * float(e.size / max(holes.sum(), 1)),
                        "ms": ms, "n_holes": int(holes.sum())}
        if keep:
            outs[name] = out
    return i, scores, outs


def _masked(job):
    """
    Masked-hole test (works without true depth, so on both datasets): the real hole shapes of frame i + 50 (wrapping) are
    punched into pixels frame i actually measured; each fill must restore those known values (errors in mm).
    """
    dataset, i, n_frames = job
    load = D.icl_noisy if dataset == "icl" else D.tum_depth
    base = median_filter(gate(load(i)))
    j = (i + min(50, n_frames // 2)) % n_frames           # another frame, same sequence
    mask = ~has_value(median_filter(gate(load(j)))) & has_value(base)
    test = base.copy()
    test[mask] = np.nan
    scores = {}
    for name, fill in FILLS.items():
        if name == "none":
            continue
        out = fill(test, (dataset, i))
        e = np.abs(out - base)[mask & has_value(out)] * 1000
        scores[name] = {"mae": float(e.mean()) if e.size else np.nan,
                        "median": float(np.median(e)) if e.size else np.nan,
                        "within_5cm": 100 * float(np.mean(e < 50)) if e.size else np.nan,
                        "filled": 100 * float(e.size / max(mask.sum(), 1)), "n": int(mask.sum())}
    return scores


def masked_scores(pool, dataset, frames, n_frames):
    res = pool.map(_masked, [(dataset, i, n_frames) for i in frames], chunksize=2)
    return {n: {k: float(np.nanmean([r[n][k] for r in res])) for k in ("mae", "median", "within_5cm", "filled", "n")}
            for n in FILLS if n != "none"}


def _masked_cols(m, n):
    if n == "none":
        return f"{'nan':>9s}{'nan':>9s}{'nan':>8s}{'0.0':>8s}"
    return f"{m[n]['mae']:9.1f}{m[n]['median']:9.1f}{m[n]['within_5cm']:8.1f}{m[n]['filled']:8.1f}"


def _filled(job):
    """One frame after gate + median + one fill method; also the fill time (ms)."""
    dataset, i, name = job
    base = median_filter(gate(D.icl_noisy(i) if dataset == "icl" else D.tum_depth(i)))
    t0 = time.perf_counter()
    out = FILLS[name](base, (dataset, i))
    return out, 1000 * (time.perf_counter() - t0)


def track_all(pool, dataset, frames, truth, camera):
    """For each fill method in turn (one sequence in memory at a time): fill, track with depth-only ICP."""
    rows, ms = {}, {}
    for name in FILLS:
        res = pool.map(_filled, [(dataset, i, name) for i in frames], chunksize=2)
        seq = [r[0] for r in res]
        ms[name] = float(np.mean([r[1] for r in res]))
        rows[name] = C.path_errors(C.track_depth(seq, camera), truth)
        print(f"  {dataset} {name:16s} ATE {rows[name]['ate_cm']:6.2f} cm", flush=True)
        del seq, res
    return rows, ms


def main():
    ap = argparse.ArgumentParser(description="Hole-filling methods after the 9x9 median: depth error and tracking")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    lines = []

    # ICL: holes scored on every frame, tracking on every step-th frame
    icl_frames = list(range(1, 1 + args.frames))
    icl_tracked = icl_frames[::args.step]
    print("ICL ...", flush=True)
    scores = {n: [] for n in FILLS}
    pool = Pool(args.jobs)
    for i, s, _ in pool.imap_unordered(_icl_frame, icl_frames, chunksize=4):
        for n in FILLS:
            scores[n].append(s[n])
    icl_track, _ = track_all(pool, "icl", icl_tracked, C.true_poses(icl_tracked), C.CAMERA)
    icl_mask = masked_scores(pool, "icl", icl_tracked, 1 + args.frames)

    n_holes = np.mean([s["n_holes"] for s in scores["none"]])
    lines += [f"ICL-NUIM kt0, frames {icl_frames[0]}-{icl_frames[-1]}, after gate + 9x9 median "
              f"(holes: {n_holes:.0f} px per frame = {100 * n_holes / (640 * 480):.2f} % of the image)",
              "hole error = |filled - true| at the real hole pixels (mm), mean over frames; "
              f"masked test = |filled - measured| at punched-in holes ({icl_mask['telea (current)']['n']:.0f} px "
              f"per frame); tracking on {len(icl_tracked)} frames",
              f"{'fill':16s}{'hole MAE':>10s}{'hole med':>10s}{'<5cm %':>8s}{'filled %':>9s}{'ms/frame':>9s}"
              f"{'ATE':>9s}{'RPE':>9s}"
              f"{'mask MAE':>9s}{'mask med':>9s}{'m<5cm %':>8s}{'m fill%':>8s}"]
    for n in FILLS:
        m = {k: np.nanmean([s[k] for s in scores[n]]) for k in ("hole_mae", "hole_median", "within_5cm", "filled", "ms")}
        d = icl_track[n]
        lines.append(f"{n:16s}{m['hole_mae']:10.1f}{m['hole_median']:10.1f}{m['within_5cm']:8.1f}{m['filled']:9.1f}"
                     f"{m['ms']:9.1f}{d['ate_cm']:9.2f}{d['rpe_mm']:9.2f}"
                     + _masked_cols(icl_mask, n))

    # TUM: tracking only (no true depth)
    tum_tracked = list(range(0, min(args.frames, D.tum_n_frames())))[::args.step]
    print("TUM ...", flush=True)
    holes_tum = 100 * np.mean([(~has_value(median_filter(gate(D.tum_depth(i))))).mean() for i in tum_tracked[::10]])
    tum_track, tms = track_all(pool, "tum", tum_tracked, tum_true_poses(tum_tracked), T.CAMERA)
    tum_mask = masked_scores(pool, "tum", tum_tracked, D.tum_n_frames())
    pool.close()
    lines += ["", f"TUM fr3 long_office_household, frames {tum_tracked[0]}-{tum_tracked[-1]} (every {args.step}-th), "
              f"after gate + 9x9 median (holes: {holes_tum:.1f} % of the image); no true depth, so fill accuracy only "
              f"from the masked test ({tum_mask['telea (current)']['n']:.0f} px per frame)",
              f"{'fill':16s}{'ms/frame':>9s}{'ATE':>9s}{'RPE':>9s}"
              f"{'mask MAE':>9s}{'mask med':>9s}{'m<5cm %':>8s}{'m fill%':>8s}"]
    for n in FILLS:
        d = tum_track[n]
        lines.append(f"{n:16s}{tms[n]:9.1f}{d['ate_cm']:9.2f}{d['rpe_mm']:9.2f}"
                     + _masked_cols(tum_mask, n))
    lines += ["", "ATE in cm, RPE (translation) in mm; lower is better except <5cm % and filled %."]

    text = "\n".join(lines) + "\n"
    print(text)
    open(os.path.join(OUT, "fills.txt"), "w").write(text)
    save_figures(scores, icl_track, tum_track, icl_mask, tum_mask)


def save_figures(scores, icl_track, tum_track, icl_mask, tum_mask):
    names = list(FILLS)
    colours = plt.cm.tab10(np.arange(len(names)))
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.4))
    x = np.arange(len(names))
    real = [np.nanmean([s["hole_mae"] for s in scores[n]]) if n != "none" else 0 for n in names]
    axs[0].bar(x - 0.27, real, 0.27, color=colours)
    axs[0].bar(x, [icl_mask[n]["mae"] if n != "none" else 0 for n in names], 0.27, color=colours, alpha=0.6, hatch="..")
    axs[0].bar(x + 0.27, [tum_mask[n]["mae"] if n != "none" else 0 for n in names], 0.27, color=colours,
               alpha=0.35, hatch="xx")
    axs[0].set_title("fill error, MAE (mm)\nsolid: ICL real holes, dotted: ICL masked, crossed: TUM masked")
    axs[0].set_xticks(x, names, rotation=30, ha="right")
    axs[1].bar(x - 0.2, [icl_mask[n]["within_5cm"] if n != "none" else 0 for n in names], 0.4, color=colours,
               hatch="..")
    axs[1].bar(x + 0.2, [tum_mask[n]["within_5cm"] if n != "none" else 0 for n in names], 0.4, color=colours,
               alpha=0.5, hatch="xx")
    axs[1].set_title("masked test: filled pixels within 5 cm (%)\ndotted: ICL, crossed: TUM")
    axs[1].set_xticks(x, names, rotation=30, ha="right")
    for ax, (title, track) in zip(axs[2:], (("ICL", icl_track), ("TUM", tum_track))):
        ax.bar(x, [track[n]["ate_cm"] for n in names], 0.6, color=colours)
        ax.set_title(f"{title}: depth-only ICP, camera path ATE (cm)")
        ax.set_xticks(x, names, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fills.png"), dpi=110)
    plt.close(fig)
    save_example(names)


def save_example(names, i=151):
    """fill_example.png: one ICL frame, zoomed on its largest hole, every fill."""
    _, s, outs = _icl_frame(i, keep=True)
    gt = D.icl_gt(i)
    hole = ~has_value(outs["none"])
    n, lab = cv2.connectedComponents(hole.astype(np.uint8))
    sizes = np.bincount(lab.ravel())[1:]
    if sizes.size:
        ys, xs = np.nonzero(lab == 1 + int(np.argmax(sizes)))
        cy, cx = int(ys.mean()), int(xs.mean())
    else:
        cy, cx = 240, 320
    h = 40
    sl = (slice(max(0, cy - h), cy + h), slice(max(0, cx - h), cx + h))
    vmin, vmax = np.nanpercentile(gt[sl], 2), np.nanpercentile(gt[sl], 98)
    fig, axs = plt.subplots(1, len(names) + 1, figsize=(2.6 * (len(names) + 1), 3))
    axs[0].imshow(gt[sl], vmin=vmin, vmax=vmax, cmap="viridis"); axs[0].set_title("true depth", fontsize=9)
    for ax, name in zip(axs[1:], names):
        ax.imshow(outs[name][sl], vmin=vmin, vmax=vmax, cmap="viridis")
        ax.contour(hole[sl], [0.5], colors="w", linewidths=0.6)
        ax.set_title(f"{name}\nhole MAE {s[name]['hole_mae']:.0f} mm" if name != "none" else "none (holes white)", fontsize=9)
    for ax in axs:
        ax.axis("off")
    fig.suptitle(f"ICL frame {i}, largest hole (white outline), after the 9x9 median", fontsize=10, y=1.06)
    fig.savefig(os.path.join(OUT, "fill_example.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
