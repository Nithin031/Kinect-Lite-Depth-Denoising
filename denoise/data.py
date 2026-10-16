"""
Read-only dataset access.

Depth convention in all of denoise/: float32 metres, NaN = no measurement.
On disk both datasets store uint16 PNGs where value / 5000 = metres and 0 = missing.

Datasets
--------
- ICL-NUIM living room kt0 (synthetic):
    lr_kt0_noisy/depth/<i>.png       filter input
    lr_kt0_noisefree/depth/<i>.png   ground truth (same camera, same frame index i)
- TUM fr3 long_office_household (real Asus Xtion, a Kinect-type sensor): no ground-truth depth exists.

Contents
--------
    png_to_m        uint16 PNG values -> metres with NaN for missing
    valid           mask of measured pixels
    icl_noisy       ICL input frame i
    icl_gt          ICL ground-truth frame i
    tum_depth       TUM depth frame i (in depth.txt order)
    tum_n_frames    number of TUM depth frames
    gate            treat depth below a minimum as missing
"""
import os

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS = os.path.join(ROOT, "outputs")

ICL_NOISY = os.path.join(ROOT, "ICL NUIM", "lr_kt0_noisy", "depth")
ICL_CLEAN = os.path.join(ROOT, "ICL NUIM", "lr_kt0_noisefree", "depth")
TUM_DIR = os.path.join(ROOT, "RGB TUM", "rgbd_dataset_freiburg3_long_office_household")

DEPTH_SCALE = 5000.0  # PNG value per metre
ICL_N_FRAMES = 1509


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def png_to_m(u16):
    """uint16 PNG values -> float32 metres; 0 becomes NaN."""
    d = u16.astype(np.float32) / np.float32(DEPTH_SCALE)
    d[u16 == 0] = np.nan
    return d


def valid(depth):
    """True where a depth value exists (finite and > 0)."""
    return np.isfinite(depth) & (depth > 0)


def _read(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if im is None or im.dtype != np.uint16:
        raise IOError(f"cannot read 16-bit depth PNG: {path}")

    return im


# ---------------------------------------------------------------------------
# ICL-NUIM
# ---------------------------------------------------------------------------

def icl_noisy(i):
    return png_to_m(_read(os.path.join(ICL_NOISY, f"{i}.png")))


def icl_gt(i):
    return png_to_m(_read(os.path.join(ICL_CLEAN, f"{i}.png")))


# ---------------------------------------------------------------------------
# TUM
# ---------------------------------------------------------------------------

_TUM = None  # cached list of depth file names, filled on first use


def _tum_list():
    """Depth file names from depth.txt (lines: timestamp filename; '#' = comment)."""
    rows = []

    with open(os.path.join(TUM_DIR, "depth.txt")) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                rows.append(line.split()[1])

    return rows


def tum_depth(i):
    global _TUM

    if _TUM is None:
        _TUM = _tum_list()

    return png_to_m(_read(os.path.join(TUM_DIR, _TUM[i])))


def tum_n_frames():
    return len(_tum_list())


# ---------------------------------------------------------------------------
# Input conditioning
# ---------------------------------------------------------------------------

def gate(depth, min_depth_m):
    """
    Treat depth below min_depth_m as missing (a Kinect cannot measure that close).
    None or 0 returns the input unchanged.
    """
    if not min_depth_m:
        return depth

    d = depth.copy()
    d[d < min_depth_m] = np.nan
    return d
