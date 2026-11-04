r"""
The paper pipeline against median filtering on ICL-NUIM living room kt0 (noisy input, known true depth and path).

    noisy depth frames -> each method -> 1. depth error against the noise-free depth
                                      -> 2. ICP between neighbouring frames -> camera path -> error against the true path

Methods:
    gate only                      the input (closer than 0.4 m = hole)
    median 9x9                     clean.median_filter after the gate: holes stay holes
    median 9x9 + fill              clean.clean: median, then inpainting
    paper (center=paper)           paper_pipeline.restore, Eq. (7) as written (i = 0 term = padded spatial mean)
    paper (center=exclude)         ... i = 0 skipped (pipeline_paper_kt.py)
    paper (center=nonlocal)        ... i = 0 searched at non-zero shifts only (pipeline_v2.py)

Trackers (--tracker):
    depth   point-to-plane ICP on the point clouds, coarse to fine: uses the depth only, so it shows the depth quality
    rgbd    Open3D hybrid RGB-D odometry (as in icp.py): depth + colour

Path errors: ATE = RMS distance between estimated and true camera positions after the best rigid alignment (cm);
RPE = RMS error of the motion between consecutive tracked frames (mm and degrees).

Run:  .venv\Scripts\python.exe src\icl_compare.py --start 1 --frames 300 --step 2 --jobs 10
Restored frames are cached in cache/icl_compare/frames/ (rerun with --redo to recompute them).
Writes src/results/icl_compare.txt, icl_camera_paths_<tracker>.png, icl_paths_<tracker>.npz (the paths),
icl_paths_<tracker>_zoom.png (zoomed views + error per frame) and icl_depth_example.png.
"""
import argparse
import os
import time
from multiprocessing import Pool

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

import data as D
import paper_pipeline as P
from clean import clean, gate, has_value, median_filter
from plot_paths import save_paths, zoom_figure

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CACHE = os.path.join(D.OUTPUTS, "icl_compare", "frames")
ICL_RGB = os.path.join(os.path.dirname(D.ICL_NOISY), "rgb")
GT_PATH = os.path.join(D.DATA_ROOT, "ICL NUIM", "gt", "livingRoom0.gt.freiburg")
CAMERA = o3d.camera.PinholeCameraIntrinsic(640, 480, 481.20, 480.00, 319.50, 239.50)   # ICL-NUIM

METHODS = {   # name -> (kind, setting)
    "gate only": ("single", gate),
    "median 9x9": ("single", lambda d: median_filter(gate(d))),
    "median 9x9 + fill": ("single", clean),
    "paper (center=paper)": ("paper", "paper"),
    "paper (center=exclude)": ("paper", "exclude"),
    "paper (center=nonlocal)": ("paper", "nonlocal"),
}


# ---------------------------------------------------------------------------
# Restoring frames (cached as 16-bit PNGs, value / 5000 = m)
# ---------------------------------------------------------------------------

def _slug(name):
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def _cache_path(name, i):
    return os.path.join(CACHE, _slug(name), f"{i}.png")


def _restore_one(job):
    """Restore frame i with one method and cache it. Runs in a worker process."""
    name, i, redo = job
    path = _cache_path(name, i)
    if os.path.exists(path) and not redo:
        return
    kind, setting = METHODS[name]
    if kind == "single":
        out = setting(D.icl_noisy(i))
    else:
        lo, hi = P.window(D.ICL_N_FRAMES, i)
        out = P.restore([D.icl_noisy(j) for j in range(lo, hi)], i - lo, center=setting)["restored"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.round(np.nan_to_num(out) * D.DEPTH_SCALE).clip(0, 65535).astype(np.uint16))


def restored(name, i):
    return D.png_to_m(cv2.imread(_cache_path(name, i), cv2.IMREAD_UNCHANGED))


# ---------------------------------------------------------------------------
# 1. Depth error against the true depth
# ---------------------------------------------------------------------------

def depth_scores(out, noisy, gt):
    """Errors in mm. 'holes' = pixels with no value after the gate (spikes and dropouts) that the method filled."""
    truth = has_value(gt)
    hole = ~has_value(gate(noisy, 0.4)) & truth
    has = has_value(out) & truth
    err = np.abs(out - gt) * 1000

    def mean(mask):
        return float(np.mean(err[mask])) if mask.any() else np.nan

    return {
        "mae": mean(has),
        "rmse": float(np.sqrt(np.mean(err[has] ** 2))),
        "median_err": float(np.median(err[has])),
        "within_1cm": 100 * float(np.mean(err[has] < 10)),
        "measured_mae": mean(has & ~hole),
        "hole_mae": mean(has & hole),
        "coverage": 100 * float(has.sum() / truth.sum()),
        "holes_filled": 100 * float((has & hole).sum() / max(hole.sum(), 1)),
    }


# ---------------------------------------------------------------------------
# 2. Tracking
# ---------------------------------------------------------------------------

def true_poses(indices):
    """
    4x4 camera-to-world pose per frame. livingRoom0.gt.freiburg rows: n tx ty tz qx qy qz qw for frames n = 1..1508
    (frame 0 has none). The file's camera y axis points the other way (ICL's fy is negative), so y is flipped.
    Checked by tracking the noise-free depth: ATE 0.1 cm with this reading, 0.7-2.9 cm with the other three.
    """
    rows = {int(float(r[0])): list(map(float, r[1:8])) for r in
            (line.split() for line in open(GT_PATH)) if r and r[0][0] != "#"}
    flip = np.diag([1.0, -1.0, 1.0, 1.0])
    poses = []
    for i in indices:
        x, y, z, qx, qy, qz, qw = rows[i]
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        pose[:3, 3] = [x, y, z]
        poses.append(flip @ pose @ flip)
    return poses


def _cloud(depth, camera=CAMERA):
    depth_u16 = np.round(np.nan_to_num(depth) * D.DEPTH_SCALE).clip(0, 65535).astype(np.uint16)
    pc = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(depth_u16), camera, depth_scale=D.DEPTH_SCALE, depth_trunc=6.0)
    return pc


def track_depth(depths, camera=CAMERA):
    """Frame-to-frame point-to-plane ICP, coarse to fine; the start guess is the previous motion."""
    scales = [(0.04, 0.10), (0.02, 0.05), (0.01, 0.02)]   # (voxel m, max matching distance m)
    clouds = []
    for d in depths:
        pc = _cloud(d, camera)
        levels = []
        for voxel, _ in scales:
            down = pc.voxel_down_sample(voxel)
            down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * voxel, max_nn=30))
            levels.append(down)
        clouds.append(levels)

    poses, motion = [np.eye(4)], np.eye(4)
    for prev, cur in zip(clouds[:-1], clouds[1:]):
        guess = motion
        for (voxel, dist), src, dst in zip(scales, cur, prev):
            guess = o3d.pipelines.registration.registration_icp(
                src, dst, dist, guess, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)).transformation
        motion = guess
        poses.append(poses[-1] @ motion)
    return poses


def track_rgbd(depths, colours, camera=CAMERA):
    """Open3D hybrid RGB-D odometry between consecutive frames (as src/icp.py)."""
    frames = []
    for c, d in zip(colours, depths):
        depth_u16 = np.round(np.nan_to_num(d) * D.DEPTH_SCALE).clip(0, 65535).astype(np.uint16)
        frames.append(o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(c), o3d.geometry.Image(depth_u16), depth_scale=D.DEPTH_SCALE, depth_trunc=6.0,
            convert_rgb_to_intensity=True))
    method = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
    poses = [np.eye(4)]
    for prev, cur in zip(frames[:-1], frames[1:]):
        ok, motion, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
            cur, prev, camera, np.eye(4), method, o3d.pipelines.odometry.OdometryOption())
        poses.append(poses[-1] @ (motion if ok else np.eye(4)))
    return poses


def path_errors(est, truth):
    """ATE (cm, after the best rigid alignment of the positions) and RPE (mm, deg) between consecutive frames."""
    rel_truth = [np.linalg.inv(truth[0]) @ t for t in truth]   # both paths start at the identity
    a = np.array([e[:3, 3] for e in est])
    b = np.array([t[:3, 3] for t in rel_truth])

    # Horn / Kabsch: rotation R, translation t minimising |R a + t - b|
    ma, mb = a.mean(0), b.mean(0)
    U, _, Vt = np.linalg.svd((a - ma).T @ (b - mb))
    S = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ S @ U.T
    ate = 100 * np.sqrt(np.mean(np.sum((a @ R.T + (mb - R @ ma) - b) ** 2, 1)))

    dt, dr = [], []
    for e0, e1, t0, t1 in zip(est[:-1], est[1:], rel_truth[:-1], rel_truth[1:]):
        err = np.linalg.inv(np.linalg.inv(t0) @ t1) @ (np.linalg.inv(e0) @ e1)
        dt.append(np.linalg.norm(err[:3, 3]))
        dr.append(np.degrees(np.arccos(np.clip((np.trace(err[:3, :3]) - 1) / 2, -1, 1))))
    return {"ate_cm": ate, "rpe_mm": 1000 * np.sqrt(np.mean(np.square(dt))),
            "rpe_deg": np.sqrt(np.mean(np.square(dr))), "aligned": a @ R.T + (mb - R @ ma), "truth": b}


# ---------------------------------------------------------------------------
# Pictures
# ---------------------------------------------------------------------------

def save_paths_figure(paths, path, title="ICL kt0 camera path, top view (aligned to the true path)"):
    plt.figure(figsize=(7, 6))
    truth = next(iter(paths.values()))["truth"]
    plt.plot(truth[:, 0], truth[:, 2], "k--", lw=2, label="true path")
    for name, r in paths.items():
        plt.plot(r["aligned"][:, 0], r["aligned"][:, 2], lw=1, label=f"{name}: ATE {r['ate_cm']:.1f} cm")
    plt.xlabel("x (m)"); plt.ylabel("z (m)"); plt.axis("equal"); plt.legend(fontsize=7)
    plt.title(title)
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()


def save_depth_figure(i, names, path):
    gt, noisy = D.icl_gt(i), D.icl_noisy(i)
    vmax = np.nanpercentile(gt, 99)
    fig, axs = plt.subplots(2, len(names), figsize=(3.4 * len(names), 5.6))
    for col, name in enumerate(names):
        d = restored(name, i)
        axs[0, col].imshow(d, cmap="viridis", vmin=0, vmax=vmax)
        axs[0, col].set_title(name, fontsize=8)
        err = np.abs(d - gt) * 1000
        axs[1, col].imshow(err, cmap="magma", vmin=0, vmax=50)
        axs[1, col].set_title(f"|error|: mean {np.nanmean(err):.1f} mm", fontsize=8)
        for ax in axs[:, col]:
            ax.axis("off")
    fig.suptitle(f"ICL kt0 frame {i}: depth (top) and |error| vs true depth, clipped at 50 mm (bottom)")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Paper pipeline vs median filter on ICL kt0: depth error and ICP")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--step", type=int, default=2, help="track every N-th frame (all frames are scored)")
    ap.add_argument("--tracker", default="both", choices=("depth", "rgbd", "both"))
    ap.add_argument("--methods", nargs="*", default=list(METHODS), help="subset of the method names")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--redo", action="store_true", help="recompute cached restored frames")
    args = ap.parse_args()

    frames = list(range(args.start, min(args.start + args.frames, D.ICL_N_FRAMES)))
    tracked = [i for i in frames[::args.step] if i >= 1]   # frame 0 has no true pose

    t0 = time.time()
    jobs = [(name, i, args.redo) for name in args.methods for i in frames]
    with Pool(args.jobs) as pool:
        for n, _ in enumerate(pool.imap_unordered(_restore_one, jobs, chunksize=2), 1):
            if n % 100 == 0 or n == len(jobs):
                print(f"restored {n}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)

    # 1. depth error
    scores = {name: [] for name in args.methods}
    for i in frames:
        noisy, gt = D.icl_noisy(i), D.icl_gt(i)
        for name in args.methods:
            scores[name].append(depth_scores(restored(name, i), noisy, gt))

    keys = ["mae", "rmse", "median_err", "within_1cm", "measured_mae", "hole_mae", "coverage", "holes_filled"]
    lines = [f"ICL-NUIM living room kt0, frames {frames[0]}-{frames[-1]} ({len(frames)} frames)", "",
             "1. Depth error against the noise-free depth (mean over frames; mm unless %)",
             f"{'method':26s}" + "".join(f"{k:>13s}" for k in keys)]
    for name in args.methods:
        row = [np.nanmean([s[k] for s in scores[name]]) for k in keys]
        lines.append(f"{name:26s}" + "".join(f"{v:13.2f}" for v in row))
    base = np.array([s["mae"] for s in scores["gate only"]]) if "gate only" in scores else None
    if base is not None:
        for name in args.methods:
            if name != "gate only":
                better = np.mean(np.array([s["mae"] for s in scores[name]]) < base) * 100
                lines.append(f"  {name}: lower MAE than the input on {better:.0f} % of frames")

    # 2. tracking
    truth = true_poses(tracked)
    colours = [cv2.cvtColor(cv2.imread(os.path.join(ICL_RGB, f"{i}.png")), cv2.COLOR_BGR2RGB) for i in tracked] \
        if args.tracker in ("rgbd", "both") else None
    sources = {"true depth (noise-free)": [D.icl_gt(i) for i in tracked]}
    sources.update({name: [restored(name, i) for i in tracked] for name in args.methods})

    for tracker in (("depth", "rgbd") if args.tracker == "both" else (args.tracker,)):
        lines += ["", f"2. Camera path with the {tracker} tracker, {len(tracked)} frames (every {args.step}-th)",
                  f"{'depth used':26s}{'ATE (cm)':>12s}{'RPE (mm)':>12s}{'RPE (deg)':>12s}"]
        paths = {}
        for name, depths in sources.items():
            est = track_depth(depths) if tracker == "depth" else track_rgbd(depths, colours)
            paths[name] = path_errors(est, truth)
            r = paths[name]
            lines.append(f"{name:26s}{r['ate_cm']:12.2f}{r['rpe_mm']:12.2f}{r['rpe_deg']:12.3f}")
            print(lines[-1], flush=True)
        os.makedirs(RESULTS, exist_ok=True)
        save_paths_figure(paths, os.path.join(RESULTS, f"icl_camera_paths_{tracker}.png"))
        save_paths(paths, tracked, os.path.join(RESULTS, f"icl_paths_{tracker}.npz"))
        zoom_figure(paths, tracked, os.path.join(RESULTS, f"icl_paths_{tracker}_zoom.png"),
                    f"ICL-NUIM living room kt0: camera paths with "
                    f"{'depth-only ICP' if tracker == 'depth' else 'RGB-D odometry'}")

    text = "\n".join(lines) + "\n"
    print(text)
    os.makedirs(RESULTS, exist_ok=True)
    open(os.path.join(RESULTS, "icl_compare.txt"), "w").write(text)
    save_depth_figure(frames[len(frames) // 2], args.methods, os.path.join(RESULTS, "icl_depth_example.png"))


if __name__ == "__main__":
    main()
