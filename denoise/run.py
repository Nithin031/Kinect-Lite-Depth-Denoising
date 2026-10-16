r"""
Run one experiment file and write its results to outputs/<name>/.

Usage
-----
    .venv\Scripts\python.exe denoise\run.py experiments\<file>.json [--frames 25:1509:50] [--workers 6]
                                            [--save-frames 275,775] [--name other_folder]

Experiment file
---------------
    {
      "name": "...",
      "dataset": "icl" | "tum",
      "frames": "start:stop:step",
      "configs": [
        {"method": "iamedian", "params": {"k": 7}, "min_depth_m": 0.4},
        {"method": "bilateral_cv2",
         "grid": {"d": [5, 9], "sigma_range_m": [0.01, 0.03]},
         "params": {"sigma_space_px": 5},
         "min_depth_m": [null, 0.4]}
      ]
    }
A "grid" expands to every combination of its values. "min_depth_m" may be one value or a list.
The unfiltered input ("raw", no gate) is always added as the reference.

Outputs
-------
    per_frame.csv               one row per frame x config
    summary.csv                 one row per config, pooled over frames
    experiment_resolved.json    the expanded config list and the exact command
    environment.json            library versions
    frames/<i>/<config>.npy     filtered depth of the --save-frames frames

A finished run is never overwritten; use --name for a new folder.

Contents
--------
    expand          experiment configs -> flat list with ids
    parse_frames    "a:b:c" -> list of frame indices
    run_frame       filter one frame with every config and compute its metrics (runs in a worker)
    summarise       per-frame rows -> one summary row per config
    main            command line entry point
"""
import argparse
import itertools
import json
import os
import platform
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D
from evaluate import BANDS, icl_frame_metrics, tum_frame_metrics
from filters import FILTERS, make_id


# ---------------------------------------------------------------------------
# Experiment set-up
# ---------------------------------------------------------------------------

def expand(configs):
    """Expand grids and gate lists into single configs; "raw" is always first."""
    out = [{"id": "raw", "method": "raw", "params": {}, "min_depth_m": None}]
    seen = {"raw"}

    for c in configs:
        grid = c.get("grid", {})
        keys = list(grid)

        gates = c.get("min_depth_m")
        if not isinstance(gates, list):
            gates = [gates]

        combos = itertools.product(*[grid[k] for k in keys]) if keys else [()]

        for values in combos:
            params = dict(c.get("params", {}))
            params.update(dict(zip(keys, values)))

            for g in gates:
                cid = make_id(c["method"], params, g)
                if cid in seen:
                    continue

                seen.add(cid)
                out.append({"id": cid, "method": c["method"], "params": params, "min_depth_m": g})

    return out


def parse_frames(spec, n):
    """"start:stop[:step]" -> list of indices, clipped to n frames."""
    start, stop, step = (spec.split(":") + ["1"])[:3]
    return list(range(int(start), min(int(stop), n), int(step)))


# ---------------------------------------------------------------------------
# Work for one frame (runs inside a worker process)
# ---------------------------------------------------------------------------

def run_frame(args):
    dataset, i, cfgs, save_dir = args

    inp = D.icl_noisy(i) if dataset == "icl" else D.tum_depth(i)

    # Filter with every config, timing each one
    outs = {}
    runtime = {}
    for c in cfgs:
        t0 = time.perf_counter()

        filt = FILTERS[c["method"]]
        outs[c["id"]] = filt(D.gate(inp, c["min_depth_m"]), **c["params"])

        runtime[c["id"]] = 1000 * (time.perf_counter() - t0)

    # Metrics
    if dataset == "icl":
        gt = D.icl_gt(i)
        metrics = icl_frame_metrics(gt, outs, D.valid(inp))
    else:
        metrics = tum_frame_metrics(inp, outs)

    # Optionally keep the filtered images for figures
    if save_dir:
        folder = os.path.join(save_dir, f"{i:05d}")
        os.makedirs(folder, exist_ok=True)

        np.save(os.path.join(folder, "input.npy"), inp)
        if dataset == "icl":
            np.save(os.path.join(folder, "gt.npy"), gt)
        for cid, o in outs.items():
            np.save(os.path.join(folder, f"{cid}.npy"), o)

    return [{"frame": i, "config": cid, "runtime_ms": runtime[cid], **m} for cid, m in metrics.items()]


# ---------------------------------------------------------------------------
# Summary over frames
# ---------------------------------------------------------------------------

def summarise(pf, cfgs, dataset):
    """
    One row per config.

    Region MAE / RMSE are pooled over all pixels of all frames (sum of errors / pixel count),
    not a mean of per-frame values. delta_vs_raw_* describe the per-frame change in all-pixel
    MAE relative to raw on the same frame.
    """
    rows = []
    raw = pf[pf.config == "raw"].set_index("frame")

    for c in cfgs:
        s = pf[pf.config == c["id"]].set_index("frame")

        r = {
            "config": c["id"],
            "method": c["method"],
            "min_depth_m": c["min_depth_m"] or 0,
            **{f"p_{k}": v for k, v in c["params"].items()},
            "frames": len(s),
            "runtime_ms_mean": s.runtime_ms.mean(),
            "runtime_ms_median": s.runtime_ms.median(),
        }

        if dataset == "icl":
            # Pooled error per region
            for reg in ["all"] + [f"band{b}" for b in BANDS] + ["flat", "flat2"]:
                n = s[f"{reg}_n"].sum()
                r[f"{reg}_mae_mm"] = 1000 * s[f"{reg}_sum_abs"].sum() / n if n else np.nan
                r[f"{reg}_rmse_mm"] = 1000 * np.sqrt(s[f"{reg}_sum_sq"].sum() / n) if n else np.nan

            # Per-frame change vs raw
            d = s.all_mae_mm - raw.all_mae_mm.reindex(s.index)
            r.update({
                "delta_vs_raw_mean_mm": d.mean(),
                "delta_vs_raw_median_mm": d.median(),
                "delta_vs_raw_std_mm": d.std(),
                "delta_vs_raw_p10_mm": d.quantile(0.1),
                "delta_vs_raw_p90_mm": d.quantile(0.9),
                "delta_vs_raw_min_mm": d.min(),
                "delta_vs_raw_max_mm": d.max(),
                "frames_worse_than_raw": int((d > 0).sum()),
                "worst_frame": int(d.idxmax()) if len(d) else -1,
            })

            # Other per-frame means, fills and edge classes
            n_filled_gt = s.filled_gt_px.sum()
            filled_mae = (s.filled_mae_mm * s.filled_gt_px).sum() / max(1, n_filled_gt) if n_filled_gt else np.nan

            r.update({
                "medae_mm_framemean": s.all_medae_mm.mean(),
                "p95_mm_framemean": s.all_p95_mm.mean(),
                "bias_mm_framemean": s.all_bias_mm.mean(),
                "own_mae_mm_framemean": s.own_mae_mm.mean(),
                "coverage_pct_mean": s.coverage_pct.mean(),
                "filled_px_total": int(s.filled_px.sum()),
                "filled_mae_mm": filled_mae,
                "bad_fills_gt5cm_total": int(s.bad_fills_gt5cm.sum()),
                "lost_px_total": int(s.lost_px.sum()),
                "edge_bad_pct": s.edge_bad_pct.mean(),
                "edge_wrong_side_pct": s.edge_wrong_side_pct.mean(),
                "edge_between_pct": s.edge_between_pct.mean(),
                "edge_outside_pct": s.edge_outside_pct.mean(),
                "matched_px_per_frame": s.matched_px.mean(),
            })

        else:
            for k in ("change_flat_mm", "change_edges_mm", "flat_roughness_mm", "holes_pct"):
                r[k] = s[k].mean()

            r["filled_px_total"] = int(s.filled_px.sum())
            r["lost_px_total"] = int(s.lost_px.sum())

        rows.append(r)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def write_run_info(out, args, exp_path, dataset, frames, cfgs):
    import cv2
    import matplotlib
    import scipy

    try:
        source = os.path.relpath(exp_path, D.ROOT)
    except ValueError:  # experiment file on another drive (Windows)
        source = os.path.abspath(exp_path)

    resolved = {
        "source": source,
        "command": " ".join(sys.argv),
        "dataset": dataset,
        "frames": frames,
        "workers": args.workers,
        "configs": cfgs,
    }
    with open(os.path.join(out, "experiment_resolved.json"), "w") as f:
        json.dump(resolved, f, indent=1)

    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out, "environment.json"), "w") as f:
        json.dump(env, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--frames", help="override start:stop:step")
    ap.add_argument("--name", help="override output folder name")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--save-frames", default="", help="comma-separated frame indices whose outputs are saved")
    args = ap.parse_args()

    with open(args.experiment) as f:
        exp = json.load(f)

    # Output folder (never overwrite a finished run)
    name = args.name or exp["name"]
    out = os.path.join(D.OUTPUTS, name)

    if os.path.exists(os.path.join(out, "summary.csv")):
        sys.exit(f"{out} already has results; use --name for a new folder (results are never overwritten)")

    os.makedirs(out, exist_ok=True)

    # Frames and configs
    dataset = exp["dataset"]
    n_frames = D.ICL_N_FRAMES if dataset == "icl" else D.tum_n_frames()
    frames = parse_frames(args.frames or exp["frames"], n_frames)
    cfgs = expand(exp["configs"])
    save = {int(x) for x in args.save_frames.split(",") if x}

    write_run_info(out, args, args.experiment, dataset, frames, cfgs)
    print(f"{name}: {dataset}, {len(frames)} frames, {len(cfgs)} configs, {args.workers} workers", flush=True)

    # Run frames in parallel
    jobs = []
    for i in frames:
        save_dir = os.path.join(out, "frames") if i in save else None
        jobs.append((dataset, i, cfgs, save_dir))

    t0 = time.time()
    rows = []
    with Pool(args.workers) as pool:
        for k, frame_rows in enumerate(pool.imap_unordered(run_frame, jobs), 1):
            rows += frame_rows
            print(f"  {k}/{len(frames)} frames  {time.time() - t0:.0f} s", flush=True)

    # Write results
    pf = pd.DataFrame(rows).sort_values(["frame", "config"])
    pf.to_csv(os.path.join(out, "per_frame.csv"), index=False)

    summary = summarise(pf, cfgs, dataset)
    summary.to_csv(os.path.join(out, "summary.csv"), index=False)

    print(f"done in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
