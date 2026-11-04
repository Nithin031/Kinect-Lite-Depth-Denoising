r"""
Does cleaning the depth help camera tracking?

    depth + colour frames -> (clean the depth) -> RGB-D odometry between neighbouring frames -> camera path
                          -> compare with the true path (ATE)

The tracker is Open3D's hybrid RGB-D odometry (Park et al., ICCV 2017). It is an ICP-style alignment that uses
BOTH the depth (geometry) and the colour image (brightness) of each frame. Depth alone slid along flat walls and
desks and was ~58 cm off; adding colour brings the error to ~4 cm, so the effect of cleaning can be measured.

Compared inputs (only the depth is cleaned, the colour image is untouched):
    raw depth (baseline, computed once and saved), gate only, gate + median, full cleaning.

Run:  .venv\Scripts\python.exe src\icp.py --start 0 --frames 200 --step 2        (--redo recomputes the baseline)
Needs the TUM dataset. Writes results.txt, camera_paths.png and cleaning_example.png into src/results/.
"""
import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

import data as D
from clean import clean, gate, median_filter

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CAMERA = o3d.camera.PinholeCameraIntrinsic(640, 480, 535.4, 539.2, 320.1, 247.6)   # TUM fr3 calibration
TRACKER = "rgbd"                     # part of the saved baseline's file name; change it when the tracker changes

CLEANING = {                         # name -> function applied to every depth image before tracking
    "raw depth (baseline)": lambda d: d,
    "gate only": gate,
    "gate + median": lambda d: median_filter(gate(d)),
    "full clean (gate + median + fill)": clean,
}


# ---------------------------------------------------------------------------
# Data: timestamps, colour images, true poses
# ---------------------------------------------------------------------------

def read_list(name):
    """(timestamp, value) rows of a TUM list file such as depth.txt or rgb.txt."""
    rows = [line.split() for line in open(os.path.join(D.TUM_DIR, name)) if line[0] != "#"]
    return [(float(r[0]), r[1:]) for r in rows]


def closest(rows, time):
    """The row whose timestamp is closest to `time`."""
    return min(rows, key=lambda r: abs(r[0] - time))


def true_poses(indices):
    """4x4 pose of the camera for each depth frame, from the ground-truth pose recorded closest in time."""
    truth = [(t, v) for t, v in read_list("groundtruth.txt")]
    depth_times = [t for t, _ in read_list("depth.txt")]

    poses = []
    for i in indices:
        x, y, z, qx, qy, qz, qw = map(float, closest(truth, depth_times[i])[1])
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        pose[:3, 3] = [x, y, z]
        poses.append(pose)
    return poses


def colour_images(indices):
    """The colour image taken closest in time to each depth frame."""
    rgb, depth_times = read_list("rgb.txt"), [t for t, _ in read_list("depth.txt")]
    files = [closest(rgb, depth_times[i])[1][0] for i in indices]
    return [cv2.cvtColor(cv2.imread(os.path.join(D.TUM_DIR, f)), cv2.COLOR_BGR2RGB) for f in files]


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

def to_rgbd(colour, depth):
    """Colour image + depth in metres (NaN = missing) -> Open3D RGB-D image."""
    depth_mm = np.round(np.nan_to_num(depth) * D.DEPTH_SCALE).clip(0, 65535).astype(np.uint16)
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(colour), o3d.geometry.Image(depth_mm),
        depth_scale=D.DEPTH_SCALE, depth_trunc=4.0, convert_rgb_to_intensity=True)


def track(colours, depths):
    """Camera path (one 4x4 pose per frame, first = identity). Each frame is aligned with the previous one."""
    frames = [to_rgbd(c, d) for c, d in zip(colours, depths)]
    method = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
    poses = [np.eye(4)]

    for previous, current in zip(frames[:-1], frames[1:]):
        ok, motion, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
            current, previous, CAMERA, np.eye(4), method, o3d.pipelines.odometry.OdometryOption())
        poses.append(poses[-1] @ (motion if ok else np.eye(4)))   # if alignment fails, assume no motion

    return poses


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

def ate_cm(estimated, truth):
    """Average distance (cm, RMS) between estimated and true camera positions, both measured from the first pose."""
    start = np.linalg.inv(truth[0])
    errors = [np.linalg.norm(e[:3, 3] - (start @ t)[:3, 3]) for e, t in zip(estimated, truth)]
    return 100 * np.sqrt(np.mean(np.square(errors)))


# ---------------------------------------------------------------------------
# Pictures
# ---------------------------------------------------------------------------

def save_pictures(raw_example, clean_example, truth, paths):
    """camera_paths.png: top view of the true and estimated paths. cleaning_example.png: one frame before and after."""
    start = np.linalg.inv(truth[0])
    plt.figure(figsize=(6, 5))
    plt.plot(*np.array([(start @ t)[[0, 2], 3] for t in truth]).T, "k--", label="true path")
    for name in ("raw depth (baseline)", "gate + median"):
        plt.plot(*np.array([p[[0, 2], 3] for p in paths[name]]).T, label=name)
    plt.xlabel("x (m)"); plt.ylabel("z (m)"); plt.axis("equal"); plt.legend(); plt.title("Camera path, top view")
    plt.savefig(os.path.join(RESULTS, "camera_paths.png"), dpi=100, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, image, title in zip(axes, [raw_example, clean_example], ["raw depth", "cleaned depth"]):
        ax.imshow(image, cmap="viridis"); ax.set_title(title); ax.axis("off")
    plt.savefig(os.path.join(RESULTS, "cleaning_example.png"), dpi=100, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="ICP-style camera tracking on raw vs cleaned depth")
    ap.add_argument("--start", type=int, default=0, help="first frame")
    ap.add_argument("--frames", type=int, default=200, help="how many frames to cover")
    ap.add_argument("--step", type=int, default=2, help="use every N-th frame")
    ap.add_argument("--redo", action="store_true", help="recompute the saved baseline")
    args = ap.parse_args()

    indices = range(args.start, args.start + args.frames, args.step)
    raw = [D.tum_depth(i) for i in indices]
    colours = colour_images(indices)
    truth = true_poses(indices)
    os.makedirs(RESULTS, exist_ok=True)

    paths, errors = {}, {}
    saved = os.path.join(RESULTS, f"baseline_{TRACKER}_{args.start}_{args.frames}_{args.step}.npy")
    for name, function in CLEANING.items():
        is_baseline = name == "raw depth (baseline)"
        if is_baseline and os.path.exists(saved) and not args.redo:
            paths[name] = list(np.load(saved))                  # baseline: computed once, then loaded
        else:
            paths[name] = track(colours, [function(d) for d in raw])
            if is_baseline:
                np.save(saved, np.array(paths[name]))
        errors[name] = ate_cm(paths[name], truth)

    text = f"{len(raw)} frames (frames {args.start}-{args.start + args.frames}, every {args.step}-th)\n"
    text += "".join(f"{name:36s}: {error:.2f} cm\n" for name, error in errors.items())
    print(text)
    open(os.path.join(RESULTS, "results.txt"), "w").write(text)

    middle = raw[len(raw) // 2]
    save_pictures(middle, clean(middle), truth, paths)


if __name__ == "__main__":
    main()
