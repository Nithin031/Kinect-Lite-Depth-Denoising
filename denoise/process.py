r"""
Process depth frames, a folder, a video or a dataset with the initial pipeline, and visualise the result.

    noisy depth -> 0.4 m gate -> denoising filter -> hole filling -> restored depth

Usage
-----
    # one depth image, a folder of depth images, or a depth video
    .venv\Scripts\python.exe denoise\process.py --input samples\tum\depth
    .venv\Scripts\python.exe denoise\process.py --input frame.png
    .venv\Scripts\python.exe denoise\process.py --input depth_video.mp4 --video-max-depth 5

    # a dataset, all frames or a range
    .venv\Scripts\python.exe denoise\process.py --dataset icl --frames 0:1509:1
    .venv\Scripts\python.exe denoise\process.py --dataset tum --frames 0:2509:5 --filter bilateral_cv2 --fill ns

Inputs
------
    depth PNG       16-bit, value / 5000 = metres, 0 = missing (TUM / ICL-NUIM format); --png-scale to change
    folder          every *.png inside, in name order (natural sort, so 2.png comes before 10.png)
    depth video     8-bit grey video: grey 0 = missing, grey 255 = --video-max-depth metres (lossy: prefer PNGs)
    dataset         icl (true depth known: error is measured) or tum

Outputs (in --out, default outputs/process/<input name>/)
-------
    restored/*.png      restored depth, same format as the input PNGs
    preview.mp4         side by side: input | denoised | restored (| error, when the true depth is known)
    frames.csv          per frame: missing pixels before / after, pixels filled, time, error (ICL)
    summary.png         those numbers along the sequence
    stages.png          every stage of one frame (--show-frame, default the middle one)
    viewer_3d.html      that frame as rotatable 3D point clouds (input, denoised, restored)

Contents
--------
    list_sources    input argument -> list of frames to load
    load            load one frame (depth in metres, true depth or None)
    process_one     run the pipeline on one frame (runs in a worker process)
    colour_frame    depth / error image -> coloured picture with a caption
    main            command line
"""
import argparse
import os
import re
import sys
import time
from multiprocessing import Pool

import cv2
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D
from pipeline import DEFAULT_FILTER, DEFAULT_FILTER_PARAMS, restore, stage_figure
from plots import INK2, SEQ, SERIES, SURFACE


VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _natural_key(name):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def list_sources(args):
    """-> (list of source items, name for the output folder). An item is whatever load() needs."""
    if args.dataset:
        n = D.ICL_N_FRAMES if args.dataset == "icl" else D.tum_n_frames()
        start, stop, step = (args.frames.split(":") + ["1"])[:3]
        frames = range(int(start), min(int(stop), n), int(step))
        return [(args.dataset, i) for i in frames], f"{args.dataset}_{args.frames.replace(':', '-')}"

    path = args.input
    name = os.path.splitext(os.path.basename(os.path.normpath(path)))[0]

    if os.path.isdir(path):
        files = sorted((f for f in os.listdir(path) if f.lower().endswith(".png")), key=_natural_key)
        if not files:
            sys.exit(f"no .png files in {path}")
        return [("png", os.path.join(path, f)) for f in files], name

    if path.lower().endswith(VIDEO_EXT):
        cap = cv2.VideoCapture(path)
        items = []
        k = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            grey = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            items.append(("video", (k, grey)))
            k += 1
        cap.release()
        if not items:
            sys.exit(f"could not read any frame from {path}")
        return items, name

    return [("png", path)], name


def _read_png_m(path, scale):
    """16-bit depth PNG -> metres, NaN = missing."""
    u16 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if u16 is None:
        raise IOError(f"cannot read {path}")
    if u16.ndim == 3:
        u16 = u16[:, :, 0]

    depth = u16.astype(np.float32) / np.float32(scale)
    depth[u16 == 0] = np.nan
    return depth


def load(item, args):
    """-> (frame name, depth in metres with NaN = missing, true depth or None)."""
    kind, ref = item

    if kind == "icl":
        return f"{ref:05d}", D.icl_noisy(ref), D.icl_gt(ref)

    if kind == "tum":
        return f"{ref:05d}", D.tum_depth(ref), None

    if kind == "png":
        depth = _read_png_m(ref, args.png_scale)

        # True depth with the same file name in --truth, if given
        gt = None
        if args.truth:
            gt_path = os.path.join(args.truth, os.path.basename(ref))
            if os.path.exists(gt_path):
                gt = _read_png_m(gt_path, args.png_scale)

        return os.path.splitext(os.path.basename(ref))[0], depth, gt

    # video frame: grey 0 = missing, 255 = max depth
    k, grey = ref
    depth = grey.astype(np.float32) / 255 * np.float32(args.video_max_depth)
    depth[grey == 0] = np.nan
    return f"{k:05d}", depth, None


# ---------------------------------------------------------------------------
# Processing (one frame per worker call)
# ---------------------------------------------------------------------------

def process_one(job):
    item, args, settings = job
    name, depth, gt = load(item, args)

    t0 = time.perf_counter()
    stages = restore(depth, **settings)
    ms = 1000 * (time.perf_counter() - t0)

    inp, gated, den, res = stages["input"], stages["gated"], stages["denoised"], stages["restored"]
    row = {
        "frame": name,
        "missing_input_pct": 100 * float((~D.valid(inp)).mean()),
        "missing_denoised_pct": 100 * float((~D.valid(den)).mean()),
        "missing_restored_pct": 100 * float((~D.valid(res)).mean()),
        "filled_px": int((D.valid(res) & ~D.valid(den)).sum()),
        "time_ms": ms,
    }

    if gt is not None:
        # Error against the true depth: input and denoised on their own valid pixels; fills on their own
        filled = D.valid(res) & ~D.valid(gated)
        row.update({
            "mae_input_mm": 1000 * float(np.nanmean(np.abs(inp - gt))),
            "mae_denoised_mm": 1000 * float(np.nanmean(np.abs(den - gt))),
            "mae_restored_mm": 1000 * float(np.nanmean(np.abs(res - gt))),
            "mae_filled_px_mm": 1000 * float(np.nanmean(np.abs(res - gt)[filled])) if filled.any() else np.nan,
        })

    return name, row, stages, gt


def write_png(depth, path, scale):
    """Metres -> 16-bit PNG with the input's scale (value / scale = metres), missing = 0."""
    u16 = np.clip(np.round(np.nan_to_num(depth, nan=0.0) * scale), 0, 65535).astype(np.uint16)
    cv2.imwrite(path, u16)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def colour_frame(img, vmax, caption, cmap=SEQ):
    """Float image -> BGR uint8 picture, missing = white, caption on a strip above."""
    norm = np.clip(np.nan_to_num(img, nan=0) / vmax, 0, 1)
    rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    rgb[~np.isfinite(img)] = 255

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    strip = np.full((28, bgr.shape[1], 3), 252, np.uint8)
    cv2.putText(strip, caption, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    return np.vstack([strip, bgr])


def video_frame(name, stages, gt, vmax):
    panels = [
        colour_frame(stages["input"], vmax, f"frame {name}: noisy input"),
        colour_frame(stages["denoised"], vmax, "denoised"),
        colour_frame(stages["restored"], vmax, "restored (holes filled)"),
    ]
    if gt is not None:
        err = np.abs(stages["restored"] - gt) * 1000
        panels.append(colour_frame(err, 50, "|error| of restored, 0-50 mm"))

    return np.hstack(panels)


def summary_figure(df, path, title):
    has_gt = "mae_input_mm" in df
    fig, axs = plt.subplots(2 if has_gt else 1, 1, figsize=(12, 7 if has_gt else 3.6), squeeze=False)
    numeric = df.frame.astype(str).str.fullmatch(r"\d+").all()
    x = df.frame.astype(int) if numeric else np.arange(len(df))

    ax = axs[0, 0]
    ax.plot(x, df.missing_input_pct, color=INK2, lw=1.2, label="input")
    ax.plot(x, df.missing_denoised_pct, color=SERIES[1], lw=1.2, label="denoised")
    ax.plot(x, df.missing_restored_pct, color=SERIES[0], lw=2, label="restored")
    ax.set_ylabel("missing pixels (%)")
    ax.set_title("Missing pixels per frame", loc="left", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

    if has_gt:
        ax = axs[1, 0]
        ax.plot(x, df.mae_input_mm, color=INK2, lw=1.2, label="noisy input")
        ax.plot(x, df.mae_denoised_mm, color=SERIES[1], lw=1.2, label="denoised")
        ax.plot(x, df.mae_restored_mm, color=SERIES[0], lw=2, label="restored (incl. filled pixels)")
        ax.set_ylabel("mean |error| (mm)")
        ax.set_title("Error against the true depth per frame", loc="left", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")

    axs[-1, 0].set_xlabel("frame number" if numeric else "frame (in processing order)")
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)


def viewer_3d(stages, camera, path, title):
    from pointcloud import INTRINSICS, backproject, css_colours, viewer

    cam = INTRINSICS[camera]
    clouds = []
    for k, (label, key) in enumerate((("noisy input", "input"), ("denoised", "denoised"), ("restored", "restored"))):
        pts, _ = backproject(stages[key], cam)
        colours = css_colours(SEQ(np.clip(pts[:, 2] / max(1e-6, np.percentile(pts[:, 2], 99)), 0, 1)))
        clouds.append((label, pts, colours, k == 0))

    viewer(path, clouds, title)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Run the initial pipeline on frames, a folder, a video or a dataset")

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="depth PNG, folder of depth PNGs, or depth video")
    src.add_argument("--dataset", choices=["icl", "tum"])
    ap.add_argument("--frames", default="0:1000000:1", help="dataset frames as start:stop:step")

    ap.add_argument("--filter", default=DEFAULT_FILTER, choices=list(DEFAULT_FILTER_PARAMS))
    ap.add_argument("--fill", default="fmm", choices=["none", "region", "fmm", "ns"])
    ap.add_argument("--gate", type=float, default=0.4, help="minimum depth in metres, 0 = off")

    ap.add_argument("--truth", help="folder with true-depth PNGs of the same file names (measures the error)")
    ap.add_argument("--png-scale", type=float, default=D.DEPTH_SCALE, help="PNG value per metre (default 5000)")
    ap.add_argument("--video-max-depth", type=float, default=5.0, help="metres at grey level 255 (video input)")
    ap.add_argument("--camera", default="tum", choices=["tum", "icl"], help="intrinsics for the 3D viewer")

    ap.add_argument("--out", help="output folder")
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--video-scale", type=float, default=1.0, help="shrink the preview video, e.g. 0.5")
    ap.add_argument("--show-frame", type=int, help="frame (position in the sequence) for stages.png and the 3D viewer")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-pngs", action="store_true", help="do not save the restored PNGs (long sequences)")
    ap.add_argument("--workers", type=int, default=4)
    return ap.parse_args()


def main():
    args = parse_args()
    items, name = list_sources(args)

    out = args.out or os.path.join(D.OUTPUTS, "process", name)
    os.makedirs(out, exist_ok=True)
    if not args.no_pngs:
        os.makedirs(os.path.join(out, "restored"), exist_ok=True)

    settings = {"filter": args.filter, "fill": args.fill, "gate_m": args.gate}
    show = args.show_frame if args.show_frame is not None else len(items) // 2
    camera = args.dataset or args.camera

    print(f"{len(items)} frame(s) -> {out}  [gate {args.gate:g} m, {args.filter}, fill {args.fill}]", flush=True)

    rows = []
    writer = None
    vmax = None
    t0 = time.time()

    with Pool(max(1, args.workers)) as pool:
        jobs = [(item, args, settings) for item in items]

        for k, (fname, row, stages, gt) in enumerate(pool.imap(process_one, jobs), 1):
            rows.append(row)
            if not args.no_pngs:
                write_png(stages["restored"], os.path.join(out, "restored", f"{fname}.png"), args.png_scale)

            # Video: one colour scale for the whole sequence, set from the first frame
            if not args.no_video:
                if vmax is None:
                    vmax = float(np.nanpercentile(stages["input"], 99))
                frame = video_frame(fname, stages, gt, vmax)
                if args.video_scale != 1.0:
                    frame = cv2.resize(frame, None, fx=args.video_scale, fy=args.video_scale, interpolation=cv2.INTER_AREA)
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(os.path.join(out, "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                             args.fps, (w, h))
                writer.write(frame)

            # One frame in detail
            if k - 1 == show:
                label = f"{args.filter}, fill {args.fill}, gate {args.gate:g} m"
                stage_figure(stages, os.path.join(out, "stages.png"), f"{name}, frame {fname}: {label}", gt)
                viewer_3d(stages, camera, os.path.join(out, "viewer_3d.html"), f"{name}, frame {fname}: {label}")

            if k % 25 == 0 or k == len(items):
                print(f"  {k}/{len(items)} frames  {time.time() - t0:.0f} s", flush=True)

    if writer is not None:
        writer.release()
        print("wrote", os.path.join(out, "preview.mp4"))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "frames.csv"), index=False)
    print("wrote", os.path.join(out, "frames.csv"))

    if len(df) > 1:
        summary_figure(df, os.path.join(out, "summary.png"),
                       f"{name}: {len(df)} frames, {args.filter}, fill {args.fill}, gate {args.gate:g} m")

    # Short report in the terminal
    print(f"\nmissing pixels: input {df.missing_input_pct.mean():.2f} % -> restored {df.missing_restored_pct.mean():.2f} %")
    print(f"time per frame: {df.time_ms.median():.0f} ms (median)")
    if "mae_input_mm" in df:
        print(f"mean error: input {df.mae_input_mm.mean():.1f} mm, denoised {df.mae_denoised_mm.mean():.1f} mm, "
              f"restored {df.mae_restored_mm.mean():.1f} mm (filled pixels {df.mae_filled_px_mm.mean():.0f} mm)")


if __name__ == "__main__":
    main()
