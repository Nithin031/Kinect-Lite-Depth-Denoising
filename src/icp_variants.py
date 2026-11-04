r"""
Which ICP variant tracks best on cleaned depth? (depth only, no colour)

    raw depth -> cleaner (none / median / paper) -> cleaned depth -> ICP variant -> camera path vs. true path

Variants (icl_compare.ICP_VARIANTS): point-to-point, point-to-plane, point-to-plane with a Tukey robust loss, generalized ICP.
All use the same frame-to-frame, coarse-to-fine setup (voxels 4/2/1 cm, matching distance 10/5/2 cm, 30 iterations).

Run:  python src/icp_variants.py --dataset both --frames 300 --step 2 --jobs 4
Restored frames are cached in cache/icp_variants/ (ICL) and cache/tum_compare/frames/ (TUM); --redo recomputes them.
Writes src/results/icp_variants.txt. ICL frames 0-300 are enough: the paper's window is cut at frame 300.
"""
import argparse
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np

import data as D
import icl_compare as C
import paper_pipeline as P
import tum_compare as T
from tum_poses import true_poses as tum_true_poses

ICL_FRAMES = 301                      # frames 0-300 exist on every machine (the Kaggle dataset has no more)
METHODS = ["gate only", "median 9x9", "paper (center=paper)"]
ICL_CACHE = os.path.join(D.OUTPUTS, "icp_variants", "icl")


def _icl_path(name, i):
    return os.path.join(ICL_CACHE, C._slug(name), f"{i}.png")


def _restore_icl(job):
    name, i, redo = job
    path = _icl_path(name, i)
    if os.path.exists(path) and not redo:
        return
    kind, setting = C.METHODS[name]
    if kind == "single":
        out = setting(D.icl_noisy(i))
    else:
        lo, hi = P.window(ICL_FRAMES, i)
        out = P.restore([D.icl_noisy(j) for j in range(lo, hi)], i - lo, center=setting)["restored"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.round(np.nan_to_num(out) * D.DEPTH_SCALE).clip(0, 65535).astype(np.uint16))


def _load(dataset, name, i):
    if name == "true depth":
        return D.icl_gt(i)
    if dataset == "icl":
        return D.png_to_m(cv2.imread(_icl_path(name, i), cv2.IMREAD_UNCHANGED))
    return T.restored(name, i)


def _track(job):
    dataset, name, variant, tracked = job
    depths = [_load(dataset, name, i) for i in tracked]
    camera = C.CAMERA if dataset == "icl" else T.CAMERA
    t0 = time.time()
    est = C.track_depth(depths, camera, icp=variant)
    truth = C.true_poses(tracked) if dataset == "icl" else tum_true_poses(tracked)
    if not np.isfinite(np.array(est)).all():            # the tracker diverged (seen with gicp on noisy depth)
        print(f"  {dataset} | {name:22s} | {variant:11s} | diverged (non-finite poses)", flush=True)
        return dataset, name, variant, float("nan"), float("nan"), float("nan")
    r = C.path_errors(est, truth)
    print(f"  {dataset} | {name:22s} | {variant:11s} | ATE {r['ate_cm']:6.2f} cm | RPE {r['rpe_mm']:6.2f} mm "
          f"({time.time() - t0:.0f} s)", flush=True)
    return dataset, name, variant, r["ate_cm"], r["rpe_mm"], r["rpe_deg"]


def main():
    ap = argparse.ArgumentParser(description="ICP variants on cleaned depth (ICL and/or TUM)")
    ap.add_argument("--dataset", default="both", choices=("icl", "tum", "both"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--variants", nargs="*", default=list(C.ICP_VARIANTS))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    datasets = ["icl", "tum"] if args.dataset == "both" else [args.dataset]
    tracked = {}
    t0 = time.time()
    with Pool(args.jobs) as pool:
        for ds in datasets:
            if ds == "icl":
                frames = list(range(max(args.start, 1), min(args.start + args.frames, ICL_FRAMES)))
                tracked[ds] = frames[::args.step]
                jobs = [(m, i, args.redo) for m in METHODS for i in tracked[ds]]
                fn = _restore_icl
            else:
                n = D.tum_n_frames()
                frames = list(range(args.start, min(args.start + args.frames, n)))
                tracked[ds] = frames[::args.step]
                jobs = [(m, i, n, args.redo) for m in METHODS for i in tracked[ds]]
                fn = T._restore_one
            for k, _ in enumerate(pool.imap_unordered(fn, jobs, chunksize=1), 1):
                if k % 50 == 0 or k == len(jobs):
                    print(f"{ds}: restored {k}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)

        jobs = [(ds, m, v, tracked[ds]) for ds in datasets
                for m in (["true depth"] if ds == "icl" else []) + METHODS for v in args.variants]
        results = pool.map(_track, jobs, chunksize=1)

    lines = []
    for ds in datasets:
        title = ("ICL-NUIM living room kt0" if ds == "icl" else "TUM RGB-D fr3 long_office_household")
        lines += [f"{title}, {len(tracked[ds])} tracked frames (every {args.step}-th); ATE (cm) / RPE (mm)", "",
                  f"{'depth used':24s}" + "".join(f"{v:>18s}" for v in args.variants)]
        for m in (["true depth"] if ds == "icl" else []) + METHODS:
            row = {r[2]: r for r in results if r[0] == ds and r[1] == m}
            lines.append(f"{m:24s}" + "".join(f"{row[v][3]:10.2f} /{row[v][4]:6.2f}" for v in args.variants))
        lines.append("")
    lines += ["Variants:"] + [f"  {k}: {v}" for k, v in C.ICP_VARIANTS.items() if k in args.variants]
    text = "\n".join(lines) + "\n"
    print(text)
    os.makedirs(C.RESULTS, exist_ok=True)
    open(os.path.join(C.RESULTS, "icp_variants.txt"), "w").write(text)


if __name__ == "__main__":
    main()
