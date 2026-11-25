r"""
LIVE STREAM DEMO: the one-screen dashboard fills in WHILE ICP is computing, one frame at a time.

    python demo/live_stream.py --start 450 --frames 30        (q = quit)

What is live: the ICP camera tracking for the three inputs (raw, median, Fu et al.), the camera path, the error graph and
the maps. What is not live: the depth cleaning (Fu et al. needs about a minute of CPU per frame); those frames were cleaned
in advance on Kaggle and sit in demo/data/cache/icl. So any start frame from 303 to 597 (minus the clip length) works.

The error shown here is the distance from the true position from the same start, with no best-fit alignment afterwards
(the video aligns the whole path after the fact, which gives smaller numbers).
"""
import argparse
import os
import sys
import time
from multiprocessing import Pipe, Process

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.dirname(HERE)
ROOT = os.path.dirname(DEMO)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "4")

import cv2
import numpy as np
import open3d as o3d

import data as D
import icl_compare as C
import dashboard as DB

METHODS = [("no cleaning", "gate_only", "#8E98A3"),
           ("median 9x9 (baseline)", "median_9x9", "#2C64A8"),
           ("Fu et al. (main)", "paper__center_paper", "#D9692A")]
CACHE = os.path.join(DEMO, "data", "cache", "icl")
SCALES = [(0.04, 0.10), (0.02, 0.05), (0.01, 0.02)]      # same coarse-to-fine ICP as the rest of the project
FIRST, LAST = 303, 597                                    # frames cleaned in advance (never used for tuning)


def cleaned(slug, i):
    return D.png_to_m(cv2.imread(os.path.join(CACHE, slug, f"{i}.png"), cv2.IMREAD_UNCHANGED))


FAST_SCALES = SCALES[:2]                                  # fast mode: skip the finest (1 cm) level


def levels(depth, scales=SCALES):
    pc = C._cloud(depth, C.CAMERA)
    out = []
    for voxel, _ in scales:
        down = pc.voxel_down_sample(voxel)
        down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * voxel, max_nn=30))
        out.append(down)
    return out


def worker(slug, conn, fast=False):
    """One tracker process. All frame numbers are sent up front; a feeder thread prepares the point clouds (normals are the
    slow part) for the coming frames while the main loop runs ICP, then sends back the camera pose of every frame in order."""
    import queue
    import threading
    scales, iters = (FAST_SCALES, 15) if fast else (SCALES, 30)
    q = queue.Queue(maxsize=6)

    def feeder():
        while True:
            i = conn.recv()
            if i is None:
                q.put(None)
                return
            q.put(levels(cleaned(slug, i), scales))

    threading.Thread(target=feeder, daemon=True).start()
    prev, pose, motion = None, np.eye(4), np.eye(4)
    while True:
        cur = q.get()
        if cur is None:
            break
        if prev is not None:
            guess = motion
            for (voxel, dist), src, dst in zip(scales, cur, prev):
                guess = o3d.pipelines.registration.registration_icp(
                    src, dst, dist, guess, C._estimator("plane", dist),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters)).transformation
            motion = guess
            pose = pose @ motion
        prev = cur
        conn.send(pose)


def main():
    ap = argparse.ArgumentParser(description="Live dashboard: ICP computed frame by frame")
    ap.add_argument("--start", type=int, default=450)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--dpi", type=float, default=100, help="100 = 1920x1080 on screen")
    ap.add_argument("--save-frames", help="headless: write every screen to this folder instead of opening a window")
    args = ap.parse_args()

    frames = list(range(args.start, args.start + args.frames))
    n = len(frames)
    if frames[0] < FIRST or frames[-1] > LAST:
        sys.exit(f"\nFrames {frames[0]}-{frames[-1]} are outside {FIRST}-{LAST}: only those were cleaned in advance "
                 f"(and never used for tuning).\nFor {args.frames} frames pick a start between {FIRST} and {LAST - args.frames + 1}.\n")
    missing = [(s, i) for _, s, _ in METHODS for i in frames if not os.path.exists(os.path.join(CACHE, s, f"{i}.png"))]
    if missing:
        sys.exit(f"\nCleaned frames missing in {CACHE} (for example {missing[0]}). Copy the Kaggle cache there first.\n")

    t0 = time.time()
    truth = C.true_poses(frames)
    rel = [np.linalg.inv(truth[0]) @ t for t in truth]
    truth_pos = np.array([r[:3, 3] for r in rel])

    # the three depth feeds, half resolution, all read from the saved cleaned frames
    depth = np.zeros((3, n, 240, 320), np.float16)
    for c, (_, slug, _) in enumerate(METHODS):
        for k, i in enumerate(frames):
            depth[c, k] = np.nan_to_num(cleaned(slug, i)[::2, ::2]).astype(np.float16)

    # map grid: sized from a few frames placed with the true poses, so it stays fixed while the clip streams in
    v, u = np.mgrid[0:240:4, 0:320:4]

    def world_xy(d, pose):
        z = np.asarray(d[::4, ::4], np.float32)
        ok = (z > 0.4) & (z < 6.0)
        p = np.stack([(u[ok] - DB.CX) * z[ok] / DB.FX, (v[ok] - DB.CY) * z[ok] / DB.FY, z[ok], np.ones(int(ok.sum()))], 1)
        return (p @ pose.T)[:, [0, 2]]

    sample = sorted(set(range(0, n, max(1, n // 8))) | {n - 1})
    pts = np.concatenate([world_xy(depth[2, j], rel[j]) for j in sample] + [truth_pos[:, [0, 2]]])
    x0, x1 = np.percentile(pts[:, 0], [0.5, 99.5]) + np.array([-0.5, 0.5])
    z0, z1 = np.percentile(pts[:, 1], [0.5, 99.5]) + np.array([-0.5, 0.5])
    nx, nz = int(np.ceil((x1 - x0) / DB.CELL)), int(np.ceil((z1 - z0) / DB.CELL))

    d0 = depth[0, 0].astype(np.float32)
    DB.RUN.update(
        labels=np.array([m[0] for m in METHODS]), colors=np.array([m[2] for m in METHODS]), tracked=np.array(frames),
        title=np.array("ICL-NUIM living room (synthetic), tracked right now"), truth=truth_pos,
        aligned=np.full((3, n, 3), np.nan, np.float32), poses=np.zeros((3, n, 4, 4), np.float32),
        err=np.full((3, n), np.nan), pos=np.zeros((3, n, 2), np.float32), depth=depth,
        maps=np.zeros((2, n, nz, nx), np.uint16), meta=np.array([x0, x1, z0, z1, 10.0]),
        range=tuple(np.percentile(d0[d0 > 0], [2, 98])))
    RUN = DB.RUN
    print(f"[setup] {n} frames ready in {time.time() - t0:.0f} s; starting 3 trackers (raw, median, Fu et al.) ...", flush=True)

    conns = []
    for _, slug, _ in METHODS:
        parent, child = Pipe()
        Process(target=worker, args=(slug, child), daemon=True).start()
        conns.append(parent)

    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)
    else:
        win = "Kinect Lite LIVE  (q = quit)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1600, 900)

    counts = [np.zeros((nz, nx), np.uint32) for _ in range(2)]
    for c in conns:
        c.send(frames[0])
    t_start = time.time()
    for k in range(n):
        tk = time.time()
        poses = [c.recv() for c in conns]                # waits for the three ICP results of frame k
        if k + 1 < n:
            for c in conns:
                c.send(frames[k + 1])                    # the trackers start the next frame while we draw this one
        for m in range(3):
            RUN["poses"][m, k] = poses[m]
            RUN["aligned"][m, k] = poses[m][:3, 3]
            RUN["pos"][m, k] = poses[m][[0, 2], 3]
            RUN["err"][m, k] = np.linalg.norm(poses[m][:3, 3] - truth_pos[k]) * 100
        for mi, m in enumerate((1, 2)):                  # maps: median depth with median poses, Fu depth with Fu poses
            xy = world_xy(depth[m, k], poses[m])
            ix, iz = ((xy[:, 0] - x0) / DB.CELL).astype(int), ((xy[:, 1] - z0) / DB.CELL).astype(int)
            ok = (ix >= 0) & (ix < nx) & (iz >= 0) & (iz < nz)
            np.add.at(counts[mi], (iz[ok], ix[ok]), 1)
            RUN["maps"][mi, k] = np.minimum(counts[mi], 65535)
        filled = counts[1][counts[1] > 0]
        if filled.size:
            RUN["meta"][4] = max(RUN["meta"][4], float(np.percentile(filled, 99)))

        fig = DB.make_fig(k, args.dpi)
        fig.canvas.draw()
        img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3], cv2.COLOR_RGB2BGR)
        DB.plt.close(fig)
        print(f"frame {frames[k]} ({k + 1}/{n}): {time.time() - tk:.1f} s | cm off: raw {RUN['err'][0, k]:.1f}, "
              f"median {RUN['err'][1, k]:.1f}, Fu {RUN['err'][2, k]:.1f}", flush=True)
        if args.save_frames:
            cv2.imwrite(os.path.join(args.save_frames, f"s{k:03d}.png"), img)
        else:
            cv2.imshow(win, img)
            if cv2.waitKey(1) in (ord("q"), 27):
                break
    print(f"[done] {n} frames in {time.time() - t_start:.0f} s ({(time.time() - t_start) / n:.1f} s per frame)", flush=True)
    for c in conns:
        c.send(None)
    if not args.save_frames:
        print("Finished. Press q in the window to close it.", flush=True)
        while cv2.waitKey(100) not in (ord("q"), 27):
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
