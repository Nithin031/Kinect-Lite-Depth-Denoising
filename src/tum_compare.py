r"""
The paper pipeline against median filtering on TUM RGB-D fr3 long_office_household: the same setup as icl_compare.py.

    depth frames -> each method -> ICP between neighbouring frames -> camera path -> error against groundtruth.txt

TUM has no true depth, so only the camera paths are compared. Methods, trackers, frame count, step and path errors
are those of icl_compare.py; only the camera, the data and the pose file differ. The true pose of each frame is the
groundtruth.txt pose recorded closest in time; the colour image is the one taken closest in time (as icp.py).

Run:  python src/tum_compare.py --start 0 --frames 300 --step 2 --jobs 4
      (set TUM_DIR to the dataset folder if it is not at "RGB TUM/rgbd_dataset_freiburg3_long_office_household")
Restored frames are cached in cache/tum_compare/frames/. Writes src/results/tum_compare.txt,
tum_camera_paths_<tracker>.png, tum_paths_<tracker>.npz (the paths) and tum_paths_<tracker>_zoom.png.
"""
import argparse
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np
import open3d as o3d

import data as D
import icl_compare as C
import paper_pipeline as P
from icp import colour_images, true_poses
from plot_paths import save_paths, zoom_figure

CAMERA = o3d.camera.PinholeCameraIntrinsic(640, 480, 535.4, 539.2, 320.1, 247.6)   # TUM fr3
CACHE = os.path.join(D.OUTPUTS, "tum_compare", "frames")


def _cache_path(name, i):
    return os.path.join(CACHE, C._slug(name), f"{i}.png")


def _restore_one(job):
    """Restore TUM frame i with one method and cache it. Runs in a worker process."""
    name, i, n_frames, redo = job
    path = _cache_path(name, i)
    if os.path.exists(path) and not redo:
        return
    kind, setting = C.METHODS[name]
    if kind == "single":
        out = setting(D.tum_depth(i))
    else:
        lo, hi = P.window(n_frames, i)
        out = P.restore([D.tum_depth(j) for j in range(lo, hi)], i - lo, center=setting)["restored"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.round(np.nan_to_num(out) * D.DEPTH_SCALE).clip(0, 65535).astype(np.uint16))


def restored(name, i):
    return D.png_to_m(cv2.imread(_cache_path(name, i), cv2.IMREAD_UNCHANGED))


def main():
    ap = argparse.ArgumentParser(description="Paper pipeline vs median filter on TUM fr3: ICP against groundtruth.txt")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--step", type=int, default=2, help="track every N-th frame")
    ap.add_argument("--tracker", default="both", choices=("depth", "rgbd", "both"))
    ap.add_argument("--methods", nargs="*", default=list(C.METHODS))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    n_frames = D.tum_n_frames()
    frames = list(range(args.start, min(args.start + args.frames, n_frames)))
    tracked = frames[::args.step]

    t0 = time.time()
    jobs = [(name, i, n_frames, args.redo) for name in args.methods for i in tracked]
    with Pool(args.jobs) as pool:
        for n, _ in enumerate(pool.imap_unordered(_restore_one, jobs, chunksize=1), 1):
            if n % 50 == 0 or n == len(jobs):
                print(f"restored {n}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)

    truth = true_poses(tracked)
    colours = colour_images(tracked) if args.tracker in ("rgbd", "both") else None
    sources = {name: [restored(name, i) for i in tracked] for name in args.methods}

    os.makedirs(C.RESULTS, exist_ok=True)
    lines = [f"TUM RGB-D fr3 long_office_household, frames {frames[0]}-{frames[-1]}"]
    for tracker in (("depth", "rgbd") if args.tracker == "both" else (args.tracker,)):
        lines += ["", f"Camera path with the {tracker} tracker, {len(tracked)} frames (every {args.step}-th), "
                      f"against groundtruth.txt",
                  f"{'depth used':26s}{'ATE (cm)':>12s}{'RPE (mm)':>12s}{'RPE (deg)':>12s}"]
        paths = {}
        for name, depths in sources.items():
            est = C.track_depth(depths, CAMERA) if tracker == "depth" else C.track_rgbd(depths, colours, CAMERA)
            r = paths[name] = C.path_errors(est, truth)
            lines.append(f"{name:26s}{r['ate_cm']:12.2f}{r['rpe_mm']:12.2f}{r['rpe_deg']:12.3f}")
            print(lines[-1], flush=True)
        C.save_paths_figure(paths, os.path.join(C.RESULTS, f"tum_camera_paths_{tracker}.png"),
                            title="TUM fr3 long_office_household camera path, top view (aligned to the true path)")
        save_paths(paths, tracked, os.path.join(C.RESULTS, f"tum_paths_{tracker}.npz"))
        zoom_figure(paths, tracked, os.path.join(C.RESULTS, f"tum_paths_{tracker}_zoom.png"),
                    f"TUM fr3 long_office_household: camera paths with "
                    f"{'depth-only ICP' if tracker == 'depth' else 'RGB-D odometry'}")

    text = "\n".join(lines) + "\n"
    print(text)
    open(os.path.join(C.RESULTS, "tum_compare.txt"), "w").write(text)


if __name__ == "__main__":
    main()
