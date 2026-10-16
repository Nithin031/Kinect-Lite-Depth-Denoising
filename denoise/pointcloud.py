r"""
3D point clouds: depth image -> points, pictures and an interactive viewer.

Usage
-----
    .venv\Scripts\python.exe denoise\pointcloud.py

Writes
------
    outputs/figures/pc_1_depth_to_cloud.png     RGB, depth image and the 3D point cloud of one ICL frame
    outputs/figures/pc_2_noise_in_3d.png        bird's-eye slice: true surface vs noisy points
    outputs/figures/pc_3_filters_in_3d.png      the same slice after one tuned filter per family
    outputs/figures/pc_4_tum_real_sensor.png    the same pictures for a real Asus Xtion frame (TUM)
    RESULTS/3d_viewer_icl.html                  rotate / zoom in a browser (needs internet for plotly.js)
    RESULTS/3d_viewer_tum.html

Back-projection (pinhole camera)
--------------------------------
A pixel (u, v) with depth Z becomes the 3D point
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = Z
in the camera frame: X right, Y down, Z forward. For plots, "up" is -Y.

Contents
--------
    INTRINSICS          fx, fy, cx, cy per dataset
    backproject         depth image -> (N, 3) points + their pixel indices
    icl_rgb, tum_rgb    colour image for a depth frame
    figure_*            the four PNG figures
    viewer              the interactive HTML viewers
"""
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D
import filters as F
from plots import FIG, INK, INK2, SEQ, SERIES, SURFACE, save


# ---------------------------------------------------------------------------
# Camera models
# ---------------------------------------------------------------------------

# ICL-NUIM lists fy = -480 (a y-up convention); the positive value gives the same geometry with Y down.
INTRINSICS = {
    "icl": {"fx": 481.20, "fy": 480.00, "cx": 319.50, "cy": 239.50},
    "tum": {"fx": 535.40, "fy": 539.20, "cx": 320.10, "cy": 247.60},   # TUM fr3 calibration
}

ICL_FRAME = 775                 # one of the frames saved by the evaluation run
TUM_FRAME = 1200
SLICE_ROWS = slice(250, 290)    # image rows used for the bird's-eye slice
N_VIEW = 40000                  # points per cloud in the 3D views and the viewer

# One tuned config per family (gate 0.4 m), same as the evaluation figures
FAMILY_PICKS = [
    ("iamedian, k 9", "iamedian", {"k": 9}),
    ("Gaussian (masked), sigma 1.5 px", "gaussian_masked", {"sigma_px": 1.5}),
    ("bilateral (masked), d 15, sigma_s 10 px, sigma_r 1 m", "bilateral_masked",
     {"d": 15, "sigma_space_px": 10, "sigma_range_m": 1.0}),
]


# ---------------------------------------------------------------------------
# Depth -> points
# ---------------------------------------------------------------------------

def backproject(depth, cam):
    """Valid pixels of a depth image (metres, NaN = missing) -> (N, 3) points and their flat pixel indices."""
    H, W = depth.shape
    v, u = np.mgrid[:H, :W]

    ok = D.valid(depth)
    z = depth[ok]

    x = (u[ok] - cam["cx"]) * z / cam["fx"]
    y = (v[ok] - cam["cy"]) * z / cam["fy"]

    points = np.stack([x, y, z], axis=1)
    index = np.flatnonzero(ok)
    return points, index


def to_plot(points):
    """Camera frame -> plot frame: x right, depth forward, up."""
    return points[:, 0], points[:, 2], -points[:, 1]


# ---------------------------------------------------------------------------
# Colour images
# ---------------------------------------------------------------------------

def icl_rgb(i):
    path = os.path.join(D.ROOT, "ICL NUIM", "lr_kt0_noisefree", "rgb", f"{i}.png")
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def tum_rgb(i):
    """Colour frame closest in time to TUM depth frame i."""
    def read_list(name):
        rows = []
        with open(os.path.join(D.TUM_DIR, name)) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    t, path = line.split()[:2]
                    rows.append((float(t), path))
        return rows

    t_depth = read_list("depth.txt")[i][0]
    rgb = read_list("rgb.txt")
    _, path = min(rgb, key=lambda r: abs(r[0] - t_depth))

    return cv2.cvtColor(cv2.imread(os.path.join(D.TUM_DIR, path)), cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def subsample(n, k, seed=0):
    """k of n indices, the same every run."""
    if n <= k:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, k, replace=False))


def cloud_3d(ax, points, colours, title, max_depth=None):
    """Oblique 3D view of a point cloud. Points farther than max_depth (default: the farthest 2 %) are not drawn,
    so a few far points do not shrink the whole view."""
    if max_depth is None:
        max_depth = np.percentile(points[:, 2], 98)
    near = points[:, 2] <= max_depth
    points = points[near]
    colours = colours[near]

    keep = subsample(len(points), N_VIEW)
    x, y, z = to_plot(points[keep])

    ax.scatter(x, y, z, c=colours[keep], s=0.4, linewidths=0)
    ax.view_init(elev=18, azim=-62)
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))

    ax.set_xlabel("x (m)", labelpad=-6)
    ax.set_ylabel("depth (m)", labelpad=-6)
    ax.set_zlabel("up (m)", labelpad=-6)
    ax.tick_params(labelsize=7, pad=-2)
    ax.set_title(title, loc="left", fontsize=9)


def slice_points(depth, cam):
    """Points from the image rows in SLICE_ROWS only (a horizontal band through the scene)."""
    band = np.full(depth.shape, np.nan, np.float32)
    band[SLICE_ROWS] = depth[SLICE_ROWS]
    return backproject(band, cam)


def birds_eye(ax, depth, cam, title, colour=SERIES[0], reference=None):
    """Seen from above: x across, depth up the page. reference = true depth drawn underneath in grey."""
    if reference is not None:
        ref, _ = slice_points(reference, cam)
        ax.scatter(ref[:, 0], ref[:, 2], s=0.8, color="#c3c2b7", linewidths=0, label="true surface (GT)")

    pts, _ = slice_points(depth, cam)
    ax.scatter(pts[:, 0], pts[:, 2], s=0.8, color=colour, linewidths=0, label="points")

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("depth from camera (m)")
    ax.set_title(title, loc="left", fontsize=9)


def depth_image(ax, depth, title, vmax):
    im = ax.imshow(depth, cmap=SEQ, vmin=0, vmax=vmax)
    ax.add_patch(plt.Rectangle((0, SLICE_ROWS.start), depth.shape[1] - 1, SLICE_ROWS.stop - SLICE_ROWS.start,
                               fill=False, edgecolor=SERIES[1], lw=1.2))
    ax.set_title(title, loc="left", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    return im


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_depth_to_cloud(gt, noisy, rgb, cam):
    fig = plt.figure(figsize=(15, 4.6))

    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(rgb)
    ax.set_title(f"1. colour image (ICL frame {ICL_FRAME})", loc="left", fontsize=9)
    ax.axis("off")

    ax = fig.add_subplot(1, 3, 2)
    im = depth_image(ax, gt, "2. depth image (orange box = rows used in the views from above)",
                     np.nanpercentile(gt, 99))
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="depth (m)")

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    pts, idx = backproject(gt, cam)
    colours = rgb.reshape(-1, 3)[idx] / 255.0
    cloud_3d(ax, pts, colours, "3. point cloud: every pixel lifted to 3D")

    fig.suptitle("From a depth image to a 3D point cloud (true depth, coloured with the RGB image)",
                 x=0.01, ha="left")
    save(fig, "pc_1_depth_to_cloud.png")


def figure_noise_in_3d(gt, noisy, cam):
    fig = plt.figure(figsize=(15, 5.2))

    # Oblique 3D: noisy cloud coloured by its error
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    pts, idx = backproject(noisy, cam)
    err = np.abs(noisy.reshape(-1)[idx] - gt.reshape(-1)[idx])
    colours = SEQ(np.clip(np.nan_to_num(err) / 0.10, 0, 1))
    cloud_3d(ax, pts, colours, "noisy cloud, colour = |error| (dark = 10 cm or more)")

    # Bird's-eye slices
    ax = fig.add_subplot(1, 3, 2)
    birds_eye(ax, gt, cam, "true depth (GT), seen from above", colour=INK)

    ax = fig.add_subplot(1, 3, 3)
    birds_eye(ax, noisy, cam, "noisy depth (blue) over the true surface (grey)", reference=gt)
    ax.legend(loc="upper right", fontsize=8, markerscale=8)

    fig.suptitle(f"What the noise does in 3D (ICL frame {ICL_FRAME}, image rows "
                 f"{SLICE_ROWS.start}-{SLICE_ROWS.stop - 1} for the views from above)", x=0.01, ha="left")
    save(fig, "pc_2_noise_in_3d.png")


def apply_picks(noisy):
    """One tuned filter per family, with the 0.4 m gate."""
    gated = D.gate(noisy, 0.4)
    return [(label, F.FILTERS[method](gated, **params)) for label, method, params in FAMILY_PICKS]


def figure_filters_in_3d(gt, noisy, cam, name, title, reference=True):
    outputs = [("raw input", noisy)] + apply_picks(noisy)

    # 2 x 2 grid, so each view stays readable at page width
    fig, axs = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True,
                            gridspec_kw={"hspace": 0.18, "wspace": 0.08})
    axs = axs.flat

    for ax, (label, depth) in zip(axs, outputs):
        birds_eye(ax, depth, cam, label, reference=gt if reference else None)
        ax.title.set_fontsize(11)

    for k, ax in enumerate(axs):
        if k % 2:
            ax.set_ylabel("")
        if k < 2:
            ax.set_xlabel("")

    if reference:
        axs[0].legend(loc="upper right", fontsize=9, markerscale=8)

    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    save(fig, name)


def figure_tum(cam):
    depth = D.tum_depth(TUM_FRAME)
    rgb = tum_rgb(TUM_FRAME)

    fig = plt.figure(figsize=(15, 4.6))

    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(rgb)
    ax.set_title(f"colour image (TUM frame {TUM_FRAME})", loc="left", fontsize=9)
    ax.axis("off")

    ax = fig.add_subplot(1, 3, 2)
    im = depth_image(ax, depth, f"depth image: {100 * np.isnan(depth).mean():.1f} % of pixels missing (blank)",
                     np.nanpercentile(depth, 99))
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="depth (m)")

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    pts, idx = backproject(depth, cam)
    cloud_3d(ax, pts, rgb.reshape(-1, 3)[idx] / 255.0, "point cloud (real Asus Xtion), points up to 4 m", max_depth=4.0)

    fig.suptitle("A real Asus Xtion frame (TUM fr3 long office household): no true depth exists",
                 x=0.01, ha="left")
    save(fig, "pc_4_tum_real_sensor.png")

    figure_filters_in_3d(None, depth, cam, "pc_4b_tum_filters_in_3d.png",
                         f"TUM frame {TUM_FRAME} seen from above (image rows {SLICE_ROWS.start}-{SLICE_ROWS.stop - 1}):\n"
                         "points after each filter; camera at (0, 0); no true depth to compare with", reference=False)


# ---------------------------------------------------------------------------
# Interactive viewer
# ---------------------------------------------------------------------------

def viewer(path, clouds, title):
    """
    clouds: list of (name, points, colours as 'rgb(r,g,b)' strings, visible at start).
    One trace per cloud; click a name in the legend to show or hide it.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    for name, pts, colours, visible in clouds:
        keep = subsample(len(pts), N_VIEW)
        x, y, z = to_plot(np.round(pts[keep], 4))

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="markers", name=name,
            marker={"size": 1.4, "color": [colours[k] for k in keep]},
            visible=True if visible else "legendonly",
            hoverinfo="skip",
        ))

    fig.update_layout(
        title=title,
        scene={"aspectmode": "data", "xaxis_title": "x (m)", "yaxis_title": "depth (m)", "zaxis_title": "up (m)",
               "camera": {"eye": {"x": -0.4, "y": -1.6, "z": 0.6}}},
        legend={"itemsizing": "constant", "title": "click to show / hide"},
        paper_bgcolor=SURFACE, margin={"l": 0, "r": 0, "t": 48, "b": 0},
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")
    print("wrote", path)


def css_colours(rgb01):
    return [f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})" for r, g, b in rgb01[:, :3]]


def error_colours(depth, gt, idx):
    err = np.abs(depth.reshape(-1)[idx] - gt.reshape(-1)[idx])
    return css_colours(SEQ(np.clip(np.nan_to_num(err) / 0.10, 0, 1)))


def viewers(gt, noisy, rgb):
    cam = INTRINSICS["icl"]

    pts, idx = backproject(gt, cam)
    clouds = [("true depth (GT), RGB colours", pts, css_colours(rgb.reshape(-1, 3)[idx] / 255.0), True)]

    pts, idx = backproject(noisy, cam)
    clouds.append(("noisy input, colour = error", pts, error_colours(noisy, gt, idx), True))

    for label, depth in apply_picks(noisy):
        pts, idx = backproject(depth, cam)
        clouds.append((f"{label}, colour = error", pts, error_colours(depth, gt, idx), False))

    viewer(os.path.join(D.ROOT, "RESULTS", "3d_viewer_icl.html"), clouds,
           f"ICL frame {ICL_FRAME}: true, noisy and filtered point clouds (light = small error, dark = 10 cm or more)")

    # TUM: colours from the RGB image (no GT, so no error colours)
    cam = INTRINSICS["tum"]
    depth = D.tum_depth(TUM_FRAME)
    rgb_t = tum_rgb(TUM_FRAME).reshape(-1, 3) / 255.0

    clouds = []
    for k, (label, d) in enumerate([("raw input", depth)] + apply_picks(depth)):
        pts, idx = backproject(d, cam)
        clouds.append((label, pts, css_colours(rgb_t[idx]), k == 0))

    viewer(os.path.join(D.ROOT, "RESULTS", "3d_viewer_tum.html"), clouds,
           f"TUM frame {TUM_FRAME} (real Asus Xtion): raw and filtered point clouds, RGB colours")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cam = INTRINSICS["icl"]
    gt = D.icl_gt(ICL_FRAME)
    noisy = D.icl_noisy(ICL_FRAME)
    rgb = icl_rgb(ICL_FRAME)

    figure_depth_to_cloud(gt, noisy, rgb, cam)
    figure_noise_in_3d(gt, noisy, cam)
    figure_filters_in_3d(gt, noisy, cam, "pc_3_filters_in_3d.png",
                         f"ICL frame {ICL_FRAME} seen from above (image rows {SLICE_ROWS.start}-{SLICE_ROWS.stop - 1}):\n"
                         "points after each filter (blue) over the true surface (grey); camera at (0, 0)")
    figure_tum(INTRINSICS["tum"])
    viewers(gt, noisy, rgb)


if __name__ == "__main__":
    main()
