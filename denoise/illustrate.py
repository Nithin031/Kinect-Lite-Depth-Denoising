r"""
Teaching figure: what each filter does to one row of depth with an edge, noise and a spike.

Usage
-----
    .venv\Scripts\python.exe denoise\illustrate.py

Writes outputs/figures/theory_filters_1d.png

The signal is made up (not dataset data): a surface at 1.0 m, a step to 2.0 m, small noise,
one spike at 0.5 m on the flat part and one near the edge. The filters are the real ones from
filters.py, run on a 15-row image made of copies of this row, so 2-D windows behave like 1-D ones.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import filters as F
from plots import INK, INK2, SERIES, save


def make_row():
    rng = np.random.default_rng(3)
    x = np.arange(120)

    true = np.where(x < 60, 1.0, 2.0).astype(np.float32)
    noisy = true + rng.normal(0, 0.01, x.size).astype(np.float32)

    noisy[30] = 0.5     # outlier on a flat surface
    noisy[57] = 0.6     # outlier next to the edge
    return x, true, noisy


def run(name, row, **params):
    image = np.repeat(row[None, :], 15, axis=0)
    return F.FILTERS[name](image, **params)[7]


def main():
    x, true, noisy = make_row()

    panels = [
        ("median, k 5: the spikes are voted out, the edge stays sharp",
         run("iamedian", noisy, k=5)),
        ("Gaussian, sigma 1.5 px: spikes are spread out, the edge is blurred",
         run("gaussian_masked", noisy, sigma_px=1.5)),
        ("bilateral, sigma_r 0.03 m: edge kept, but the spikes are kept too",
         run("bilateral_masked", noisy, d=9, sigma_space_px=5, sigma_range_m=0.03)),
        ("bilateral, sigma_r 1 m: acts like a Gaussian, spikes spread",
         run("bilateral_masked", noisy, d=9, sigma_space_px=5, sigma_range_m=1.0)),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(13, 6.4), sharex=True, sharey=True,
                            gridspec_kw={"hspace": 0.32, "wspace": 0.06})

    for ax, (title, out) in zip(axs.flat, panels):
        ax.plot(x, true, color=INK2, lw=1, ls="--", label="true depth")
        ax.plot(x, noisy, ".", color=INK2, ms=4, label="noisy input")
        ax.plot(x, out, color=SERIES[0], lw=2, label="filter output")
        ax.set_title(title, loc="left", fontsize=9)

    for ax in axs[:, 0]:
        ax.set_ylabel("depth (m)")
    for ax in axs[1]:
        ax.set_xlabel("pixel along the row")

    axs[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Illustration (made-up row): a depth step at pixel 60 and two spikes, through four filters",
                 x=0.01, ha="left")
    save(fig, "theory_filters_1d.png")


if __name__ == "__main__":
    main()
