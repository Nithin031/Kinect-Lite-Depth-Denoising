r"""
DASHBOARD: everything on one screen, no introduction.

    1 raw camera feed (RGB)   2 raw depth map   3 depth after Fu et al.
    4 camera movement (top view, true path vs each method)
    5 ICP tracking error over time
    6 the maps ICP builds from its camera poses (median vs Fu et al.)

    # a video from a saved run (this is what runs on Kaggle):
    python demo/dashboard.py --run demo/data/results/run_icl_303 --video demo/videos/video_dashboard_icl.mp4

    # laptop demo: ICP runs live on the frames you pick, then the dashboard plays (q = quit):
    python demo/dashboard.py --live --start 303 --frames 60

A run is saved by:  python demo/live_demo.py --data icl --start 303 --frames 60 --step 1 --save --no-window
(ICL-NUIM only: it is the dataset with a raw RGB feed next to the depth.)
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.dirname(HERE)
ROOT = os.path.dirname(DEMO)
RGB_DIR = os.path.join(ROOT, "ICL NUIM", "lr_kt0_noisy", "rgb")
INK, MUTED, ORANGE, BLUE, GREY = "#15202B", "#4A5563", "#D9692A", "#2C64A8", "#8E98A3"
FX, FY, CX, CY = 240.6, 240.0, 159.5, 119.5      # ICL-NUIM intrinsics at the half resolution stored in the run
CELL = 0.02                                       # map cell size in metres
RUN = {}


# ------------------------------------------------------------------------------------------------ maps built by ICP
def build_maps(base):
    """Accumulate the points of every frame, moved by the ICP poses, into a top-view count image (one per frame)."""
    out = base + "_maps.npy"
    if os.path.exists(out) and os.path.exists(base + "_maps_meta.npy"):
        return
    run = np.load(base + ".npz", allow_pickle=True)
    depth = np.load(base + "_depth.npy", mmap_mode="r")
    poses = run["poses"]
    n = depth.shape[1]
    v, u = np.mgrid[0:240:4, 0:320:4]
    pts = {}
    for m in (1, 2):                                          # median, Fu et al.
        frames = []
        for k in range(n):
            z = np.asarray(depth[m, k, ::4, ::4], np.float32)
            ok = (z > 0.4) & (z < 6.0)
            p = np.stack([(u[ok] - CX) * z[ok] / FX, (v[ok] - CY) * z[ok] / FY, z[ok], np.ones(int(ok.sum()))], 1)
            frames.append((p @ poses[m, k].T)[:, [0, 2]])
        pts[m] = frames
    allp = np.concatenate([f for m in pts for f in pts[m]])
    x0, x1 = np.percentile(allp[:, 0], [0.5, 99.5]) + np.array([-0.2, 0.2])
    z0, z1 = np.percentile(allp[:, 1], [0.5, 99.5]) + np.array([-0.2, 0.2])
    cam = poses[1:, :, [0, 2], 3]                                  # keep every camera position inside the map
    x0, x1 = min(x0, cam[..., 0].min() - 0.15), max(x1, cam[..., 0].max() + 0.15)
    z0, z1 = min(z0, cam[..., 1].min() - 0.15), max(z1, cam[..., 1].max() + 0.15)
    nx, nz = int(np.ceil((x1 - x0) / CELL)), int(np.ceil((z1 - z0) / CELL))
    maps = np.zeros((2, n, nz, nx), np.uint16)
    for mi, m in enumerate((1, 2)):
        counts = np.zeros((nz, nx), np.uint32)
        for k in range(n):
            xy = pts[m][k]
            ix, iz = ((xy[:, 0] - x0) / CELL).astype(int), ((xy[:, 1] - z0) / CELL).astype(int)
            ok = (ix >= 0) & (ix < nx) & (iz >= 0) & (iz < nz)
            np.add.at(counts, (iz[ok], ix[ok]), 1)
            maps[mi, k] = np.minimum(counts, 65535)
    final = maps[:, -1][maps[:, -1] > 0]
    np.save(out, maps)
    np.save(base + "_maps_meta.npy", np.array([x0, x1, z0, z1, float(np.percentile(final, 99))]))


# ------------------------------------------------------------------------------------------------ drawing
def init(base):
    """Worker setup: small arrays in memory, big arrays memory-mapped so workers share them."""
    run = np.load(base + ".npz", allow_pickle=True)
    RUN.update({k: run[k] for k in run.files})
    RUN["depth"] = np.load(base + "_depth.npy", mmap_mode="r")
    RUN["maps"] = np.load(base + "_maps.npy", mmap_mode="r")
    RUN["meta"] = np.load(base + "_maps_meta.npy")
    d0 = np.asarray(RUN["depth"][0, 0], np.float32)
    RUN["range"] = tuple(np.percentile(d0[d0 > 0], [2, 98]))
    RUN["err"] = np.linalg.norm(RUN["aligned"] - RUN["truth"][None], axis=2) * 100      # cm, per method per frame
    RUN["pos"] = RUN["poses"][:, :, [0, 2], 3]                                         # camera position (x, z) per frame


def panel(ax, title, color=INK, border=None):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=19, weight="bold", color=color, pad=9)
    for s in ax.spines.values():
        s.set_color(border or "#C9CED4"); s.set_linewidth(5 if border else 1.5)


def make_fig(k, dpi):
    """The whole six-panel screen for frame k, drawn from RUN. Returns the matplotlib figure."""
    labels, colors, tr = list(RUN["labels"]), list(RUN["colors"]), RUN["tracked"]
    n = len(tr)
    lo, hi = RUN["range"]
    fig = plt.figure(figsize=(19.2, 10.8), dpi=dpi, facecolor="white")
    fig.text(0.015, 0.955, "LIVE  ·  " + str(RUN["title"]), fontsize=25, weight="bold", color=INK)
    fig.text(0.985, 0.955, f"frame {tr[k]}   ({k + 1}/{n})", fontsize=25, weight="bold", color=ORANGE, ha="right")

    # ---- row 1: the three feeds -------------------------------------------------------------------------------
    rgb = cv2.cvtColor(cv2.imread(os.path.join(RGB_DIR, f"{tr[k]}.png")), cv2.COLOR_BGR2RGB)
    ax = fig.add_axes([0.015, 0.50, 0.31, 0.413]); ax.imshow(rgb, interpolation="nearest")
    panel(ax, "1  Raw camera feed (RGB)")
    for c, title, col, border in ((0, "2  Raw depth map (sensor noise)", GREY, None), (1, "3  Depth after median cleaning", BLUE, BLUE)):
        ax = fig.add_axes([0.345 + (0.3275 if c == 1 else 0.0), 0.50, 0.31, 0.413])
        d = np.asarray(RUN["depth"][c, k], np.float32)
        ax.imshow(np.where(d > 0, d, np.nan), cmap="viridis", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_facecolor(INK); panel(ax, title, col, border)
    fig.text(0.015, 0.478, "Depth colour: blue = near, yellow = far.  Dark specks in 2 are spikes and holes. The median (3) gives the cleanest depth image, yet Fu et al. gives the best camera path (4, 5).",
             fontsize=14, color=MUTED)

    # ---- row 2: movement, ICP error, maps ---------------------------------------------------------------------
    A, t = RUN["aligned"], RUN["truth"]
    ax = fig.add_axes([0.05, 0.085, 0.235, 0.345])
    for c in range(3):
        ax.plot(A[c, :k + 1, 0], A[c, :k + 1, 2], color=colors[c], lw=4.5 if c == 2 else 2.8,
                label=f"{labels[c].split(' (')[0]}: {RUN['err'][c, k]:.1f} cm off")
        ax.plot(A[c, k, 0], A[c, k, 2], "o", color=colors[c], ms=11)
    ax.plot(t[:k + 1, 0], t[:k + 1, 2], "--", color=INK, lw=2.6, label="true path")
    ax.plot(t[k, 0], t[k, 2], "*", color=INK, ms=18)
    xs = np.concatenate([A[:, :k + 1, 0].ravel(), t[:k + 1, 0]]); zs = np.concatenate([A[:, :k + 1, 2].ravel(), t[:k + 1, 2]])
    half = max(np.ptp(xs), np.ptp(zs), 0.2) / 2 * 1.2
    ax.set_xlim(xs.mean() - half, xs.mean() + half); ax.set_ylim(zs.mean() - half, zs.mean() + half)
    ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=0.3); ax.tick_params(labelsize=12)
    ax.set_title("4  Camera movement (top view)", fontsize=19, weight="bold", pad=9)
    ax.legend(fontsize=12, loc="upper left", framealpha=0.9)
    ax.set_xlabel("x (m)", fontsize=13); ax.set_ylabel("z (m)", fontsize=13)

    ax = fig.add_axes([0.345, 0.085, 0.235, 0.345])
    for c in range(3):
        ax.semilogy(np.arange(k + 1), np.maximum(RUN["err"][c, :k + 1], 0.01), color=colors[c], lw=4 if c == 2 else 2.5)
        ax.plot(k, max(RUN["err"][c, k], 0.01), "o", color=colors[c], ms=10)
    ax.set_xlim(0, n - 1); ax.set_ylim(max(0.05, np.nanmin(RUN["err"][2]) * 0.7), np.nanmax(RUN["err"]) * 1.6)
    ax.grid(alpha=0.3, which="both"); ax.tick_params(labelsize=12)
    ax.set_title("5  ICP tracking error", fontsize=19, weight="bold", pad=9)
    ax.set_xlabel("frame in clip", fontsize=13); ax.set_ylabel("distance from true position (cm, log)", fontsize=13)

    x0, x1, z0, z1, vmax = RUN["meta"]
    for i, (mi, title, col, lbl) in enumerate(((0, "6a  ICP map, median", BLUE, 1), (1, "6b  ICP map, Fu et al.", ORANGE, 2))):
        ax = fig.add_axes([0.612 + i * 0.19, 0.085, 0.175, 0.345])
        img = np.log1p(np.asarray(RUN["maps"][mi, k], np.float32))
        ax.imshow(img, origin="lower", extent=[x0, x1, z0, z1], cmap="magma", vmin=0, vmax=np.log1p(vmax), interpolation="nearest")
        ax.plot(*RUN["pos"][lbl, k], "o", color="#00E5FF", ms=9, mec="white")
        ax.set_xlim(x0, x1); ax.set_ylim(z0, z1)
        panel(ax, title, col, ORANGE if lbl == 2 else None)
    fig.text(0.612, 0.045, "Every frame's points placed with ICP's poses (top view).\nBlurred or doubled walls = ICP drifted.  Cyan dot = camera.",
             fontsize=13, color=MUTED, va="center")
    return fig


def draw(task):
    """Draw frame k and save it as an image (used for the video and the replay)."""
    k, path, dpi = task
    fig = make_fig(k, dpi)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------------------------ video / live
def render_frames(base, out_dir, dpi, jobs):
    build_maps(base)
    n = len(np.load(base + ".npz", allow_pickle=True)["tracked"])
    tasks = [(k, os.path.join(out_dir, f"u{k:05d}.png"), dpi) for k in range(n)]
    t0 = time.time()
    with Pool(jobs, initializer=init, initargs=(base,)) as pool:
        for d, _ in enumerate(pool.imap(draw, tasks, chunksize=2), 1):
            if d % 20 == 0 or d == n:
                print(f"   rendered {d}/{n} frames ({time.time() - t0:.0f} s)", flush=True)
    return [t[1] for t in tasks]


def make_video(frames, out, fps):
    tmp = os.path.dirname(frames[0])
    seq = frames + [frames[-1]] * int(5 * fps)                    # hold the last frame for 5 s, nothing else added
    for j, p in enumerate(seq):
        shutil.copy(p, os.path.join(tmp, f"f{j:06d}.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "f%06d.png"),
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "30", "-vf", "scale=2560:1440", out], check=True)
    print(f"video: {out}  ({len(seq) / fps:.0f} s)", flush=True)


def play(frames, fps):
    win = "Kinect Lite live demo  (q = quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1600, 900)
    imgs = [cv2.imread(p) for p in frames]
    while True:
        for i, img in enumerate(imgs):
            cv2.imshow(win, img)
            if cv2.waitKey(int(1000 / fps) + (1500 if i == len(imgs) - 1 else 0)) in (ord("q"), 27):
                cv2.destroyAllWindows()
                return


def main():
    ap = argparse.ArgumentParser(description="One-screen dashboard: raw feed, depth, camera movement, ICP")
    ap.add_argument("--run", help="saved run, e.g. demo/data/results/run_icl_303 (no extension)")
    ap.add_argument("--live", action="store_true", help="run the pipeline now on --start/--frames, then play")
    ap.add_argument("--start", type=int, default=303)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--video", help="write an mp4 (2560x1440)")
    ap.add_argument("--fps", type=float, default=None, help="playback speed in frames per second")
    ap.add_argument("--jobs", type=int, default=max(1, min(6, os.cpu_count() or 2)))
    args = ap.parse_args()

    if args.live:
        print(f"[live] ICP on ICL-NUIM frames {args.start}-{args.start + args.frames - 1}, every frame ...", flush=True)
        subprocess.run([sys.executable, os.path.join(HERE, "live_demo.py"), "--data", "icl", "--start", str(args.start),
                        "--frames", str(args.frames), "--step", "1", "--save", "--no-window"], check=True)
        args.run = os.path.join(DEMO, "data", "results", f"run_icl_{args.start}")
    if not args.run:
        sys.exit("give --run <saved run> or --live")
    base = os.path.abspath(args.run)
    fps = args.fps or (6 if args.live else 4)
    dpi = 400 / 3 if args.video else 100                           # 2560x1440 for the video, 1920x1080 on screen
    tmp = tempfile.mkdtemp(prefix="kl_dash_")
    print("[render] drawing the dashboard frames ...", flush=True)
    frames = render_frames(base, tmp, dpi, args.jobs)
    if args.video:
        make_video(frames, os.path.abspath(args.video), fps)
    if args.live or not args.video:
        print("[play] press q in the window to quit", flush=True)
        play(frames, fps)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
