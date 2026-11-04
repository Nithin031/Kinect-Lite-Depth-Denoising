"""
Depth cleaning in 3 simple steps (the default pipeline, same result as pipeline.restore()).

    noisy depth -> 1. gate -> 2. median filter -> 3. fill holes -> clean depth

Depth is a float32 image in metres. NaN means "no measurement" (a hole).
"""
import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage


def has_value(depth):
    return np.isfinite(depth) & (depth > 0)


def gate(depth, min_m=0.4):
    """Step 1: a Kinect cannot measure closer than 0.4 m, so anything closer is wrong -> make it a hole."""
    out = depth.copy()
    out[depth < min_m] = np.nan
    return out


def median_filter(depth, k=9):
    """
    Step 2: replace every measured pixel by the median of the measured pixels in its k x k window.
    Holes are ignored (not counted as 0), and holes stay holes. Median removes isolated noisy spikes.
    """
    ok = has_value(depth)
    r = k // 2

    # collect the k*k neighbours of every pixel; holes and outside-of-image become +inf
    padded = np.pad(np.where(ok, depth, np.inf), r, constant_values=np.inf)
    windows = sliding_window_view(padded, (k, k)).reshape(*depth.shape, k * k)

    count = np.isfinite(windows).sum(-1)          # how many real values are in each window
    windows = np.sort(windows, axis=-1)           # real values first, +inf at the end

    # middle of the real values (average the two middle ones if the count is even)
    low = np.take_along_axis(windows, ((count - 1) // 2)[..., None], -1)[..., 0]
    high = np.take_along_axis(windows, (count // 2)[..., None], -1)[..., 0]

    out = depth.copy()
    out[ok] = (0.5 * (low + high))[ok]
    return out


def fill_holes(depth, radius=5):
    """
    Step 3: fill every hole from its surroundings (OpenCV inpainting, Telea / FMM method).
    The filled value is kept between the min and max depth of the measured pixels around that hole,
    because inpainting can overshoot.
    """
    ok = has_value(depth)
    n, hole_id = cv2.connectedComponents((~ok).astype(np.uint8), connectivity=8)   # label each hole 1..n-1
    if n == 1:
        return depth.copy()

    filled = cv2.inpaint(np.nan_to_num(depth).astype(np.float32), (~ok).astype(np.uint8), radius, cv2.INPAINT_TELEA)

    # measured pixels touching a hole take that hole's id -> their min/max depth is the allowed range
    ring = ndimage.grey_dilation(hole_id, size=(3, 3)) * ok
    ids = np.arange(n)
    low = np.asarray(ndimage.minimum(depth, ring, ids), np.float32)
    high = np.asarray(ndimage.maximum(depth, ring, ids), np.float32)

    out = depth.copy()
    hole = ~ok
    out[hole] = np.clip(filled[hole], low[hole_id][hole], high[hole_id][hole])
    return out


def clean(depth):
    """Run all 3 steps."""
    return fill_holes(median_filter(gate(depth)))
