r"""
Camera-path figures that show how far apart the methods' paths are, not only the overall view.

    full top view  |  zoom A: where all paths spread most  |  zoom B: where the paths near the truth spread most
    ---------------------------------------------------------------------------------------------------
    position error of every method, frame by frame (log scale)

The compare scripts (icl_compare.py, tum_compare.py) save each tracker's paths to
src/results/<dataset>_paths_<tracker>.npz and draw this figure. To redraw from the saved paths alone
(no tracking), run:

    python src/plot_paths.py            # every *_paths_*.npz in src/results/
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
REFERENCE = {"true depth (noise-free)"}

# One fixed look per method, so every figure uses the same colours. The three paper readings give almost
# identical paths: they share one colour family and differ by line style, so overlapping lines stay visible.
STYLE = {
    "true depth (noise-free)": dict(color="0.55", ls="-", lw=1.6),
    "gate only":               dict(color="tab:blue", ls="-", lw=1.4),
    "median 9x9":              dict(color="tab:orange", ls="-", lw=1.8),
    "median 9x9 + fill":       dict(color="tab:green", ls="-", lw=1.4),
    "paper (center=paper)":    dict(color="tab:red", ls="-", lw=2.4),
    "paper (center=exclude)":  dict(color="tab:purple", ls="--", lw=1.6),
    "paper (center=nonlocal)": dict(color="tab:brown", ls=":", lw=1.6),
}


def save_paths(paths, frames, file):
    """paths: name -> path_errors() result (needs 'aligned', 'truth', 'ate_cm', 'rpe_mm', 'rpe_deg')."""
    names = list(paths)
    np.savez(file, names=np.array(names), frames=np.asarray(frames),
             truth=next(iter(paths.values()))["truth"],
             aligned=np.stack([paths[n]["aligned"] for n in names]),
             ate_cm=np.array([paths[n]["ate_cm"] for n in names]),
             rpe_mm=np.array([paths[n]["rpe_mm"] for n in names]),
             rpe_deg=np.array([paths[n]["rpe_deg"] for n in names]))


def load_paths(file):
    z = np.load(file)
    paths = {str(n): {"aligned": a, "truth": z["truth"], "ate_cm": float(e)}
             for n, a, e in zip(z["names"], z["aligned"], z["ate_cm"])}
    return paths, z["frames"]


def _box(paths, names, k, half_window, min_half=0.02):
    """Square x/z limits around frames k - half_window .. k + half_window of the given paths and the truth."""
    truth = next(iter(paths.values()))["truth"]
    lo, hi = max(0, k - half_window), min(len(truth), k + half_window + 1)
    pts = np.concatenate([truth[lo:hi]] + [paths[n]["aligned"][lo:hi] for n in names])[:, [0, 2]]
    centre = (pts.min(0) + pts.max(0)) / 2
    half = max(0.6 * (pts.max(0) - pts.min(0)).max(), min_half)
    return centre[0] - half, centre[0] + half, centre[1] - half, centre[1] + half


def _spread_frame(errors, names, avoid=None, min_gap=0):
    """Frame index where the position errors of the given methods differ most (max - min)."""
    e = np.stack([errors[n] for n in names])
    spread = e.max(0) - e.min(0)
    if avoid is not None:
        spread[max(0, avoid - min_gap):avoid + min_gap + 1] = -1
    return int(np.argmax(spread))


def _draw(ax, paths, truth, frames_slice=slice(None)):
    ax.plot(truth[frames_slice, 0], truth[frames_slice, 2], "k--", lw=2.2, label="true path", zorder=10)
    for name, r in paths.items():
        a = r["aligned"][frames_slice]
        ax.plot(a[:, 0], a[:, 2], label=f"{name} (ATE {r['ate_cm']:.2f} cm)", **STYLE.get(name, {}))


def zoom_figure(paths, frames, out, title):
    truth = next(iter(paths.values()))["truth"]
    errors = {n: np.linalg.norm(r["aligned"] - truth, axis=1) for n, r in paths.items()}
    names = list(paths)
    # the paths close to the truth: within 2x the best denoised path's ATE (plus the noise-free reference)
    best_ate = min(paths[n]["ate_cm"] for n in names if n not in REFERENCE)
    close = [n for n in names if n in REFERENCE or paths[n]["ate_cm"] <= 2 * best_ate]
    if len(close) < 2:
        close = sorted(names, key=lambda n: paths[n]["ate_cm"])[:2]
    n = len(truth)
    half_window = max(4, n // 25)

    k_a = _spread_frame(errors, names)
    k_b = _spread_frame(errors, close, avoid=k_a, min_gap=n // 6)
    boxes = {"A": (k_a, _box(paths, names, k_a, half_window)), "B": (k_b, _box(paths, close, k_b, half_window))}

    fig = plt.figure(figsize=(15, 10.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.28, wspace=0.22)

    ax = fig.add_subplot(gs[0, 0])
    _draw(ax, paths, truth)
    for label, (k, (x0, x1, z0, z1)) in boxes.items():
        ax.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0, fill=False, ec="k", lw=1.2, zorder=20))
        ax.text(x1, z1, f" {label}", fontsize=12, weight="bold", va="bottom", zorder=20)
    ax.set_title("full path (top view)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)"); ax.set_aspect("equal", adjustable="datalim")

    descr = {"A": "zoom A: where all paths spread most",
             "B": "zoom B: where the paths closest to the truth spread most"}
    for col, (label, (k, (x0, x1, z0, z1))) in zip((1, 2), boxes.items()):
        ax = fig.add_subplot(gs[0, col])
        seg = slice(max(0, k - 2 * half_window), min(n, k + 2 * half_window + 1))   # only this stretch
        _draw(ax, paths, truth, seg)
        for name in names:
            p = paths[name]["aligned"][k]
            ax.plot(p[0], p[2], "o", ms=5, color=STYLE.get(name, {}).get("color"), mec="k", mew=0.5, zorder=15)
        ax.plot(truth[k, 0], truth[k, 2], "k*", ms=12, zorder=16)
        ax.set_xlim(x0, x1); ax.set_ylim(z0, z1); ax.set_aspect("equal")
        ax.xaxis.set_major_locator(MaxNLocator(5)); ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_title(f"{descr[label]}\nframes {frames[seg][0]}-{frames[seg][-1]}\n"
                     f"dots: every path at frame {frames[k]}; star: true position", fontsize=10)
        ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")

    ax = fig.add_subplot(gs[1, :])
    for name in names:
        ax.plot(frames, np.maximum(errors[name] * 100, 1e-3),
                label=f"{name} (ATE {paths[name]['ate_cm']:.2f} cm)", **STYLE.get(name, {}))
    for label, (k, _) in boxes.items():
        ax.axvline(frames[k], color="k", lw=0.8, ls=":")
        ax.text(frames[k], 1.01, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom", weight="bold")
    ax.set_yscale("log")
    ax.set_xlabel("frame"); ax.set_ylabel("position error (cm, log scale)")
    ax.set_title("distance from the true camera position, frame by frame (after aligning each whole path)", pad=16)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)

    fig.suptitle(title, fontsize=14)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    titles = {"icl": "ICL-NUIM living room kt0", "tum": "TUM fr3 long_office_household"}
    trackers = {"depth": "depth-only ICP", "rgbd": "RGB-D odometry"}
    files = sorted(glob.glob(os.path.join(RESULTS, "*_paths_*.npz")))
    if not files:
        print("no saved paths in", RESULTS, "- run icl_compare.py / tum_compare.py first")
    for file in files:
        dataset, _, tracker = os.path.splitext(os.path.basename(file))[0].split("_")
        paths, frames = load_paths(file)
        out = file.replace(".npz", "_zoom.png")
        zoom_figure(paths, frames, out, f"{titles.get(dataset, dataset)}: camera paths with {trackers.get(tracker, tracker)}")
        print("wrote", out)


if __name__ == "__main__":
    main()
