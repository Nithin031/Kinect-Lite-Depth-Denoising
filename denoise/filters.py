"""
Spatial depth filters.

Conventions
-----------
- Input and output depth are float32 metres, NaN = missing.
- Parameters are in the unit their name says (px or m). Nothing is rescaled inside a filter.

Two ways a filter can treat missing pixels
------------------------------------------
- "cv2" / plain versions: the library needs a number everywhere, so missing pixels go in as 0 m.
  A 0 m neighbour can pull nearby values towards 0.
- "masked" / invalid-aware versions: missing pixels are left out of the window entirely.

Contents
--------
    raw                 unchanged copy (the reference)
    median              plain k x k median
    iamedian            invalid-aware median
    gaussian_cv2        OpenCV Gaussian blur
    gaussian_masked     normalised Gaussian (valid pixels only)
    bilateral_cv2       OpenCV bilateral filter
    bilateral_masked    normalised bilateral (valid pixels only)
    hole_fill           fills small interior holes, never changes measured pixels
    inpaint             fills holes with OpenCV inpainting: FMM (Telea) or Navier-Stokes

    pipeline            the initial pipeline: denoising filter, then hole filling (noisy -> restored depth)

    FILTERS             name -> function, used by run.py
    make_id             builds a config name such as "bilateral_cv2_d9_s5_r0.03_gate0.4"
"""
import warnings

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage

from data import valid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(out):
    """Return float32 with every non-positive or non-finite value set to NaN (missing)."""
    out = out.astype(np.float32)
    out[~valid(out)] = np.nan
    return out


def _valid_window_median(z, ok, k):
    """
    Median of the valid pixels in each k x k window.

    Returns (median, n_valid). Missing pixels are replaced by +inf so that sorting pushes
    them to the end of each window; the median is then taken over the first n_valid values.
    Pixels outside the image count as missing.
    """
    r = k // 2

    padded = np.pad(np.where(ok, z, np.inf), r, mode="constant", constant_values=np.inf)
    windows = sliding_window_view(padded, (k, k)).reshape(z.shape[0], z.shape[1], -1)

    n_valid = np.isfinite(windows).sum(-1)
    ordered = np.sort(windows, axis=-1)

    # Middle element(s) of the valid part; for an even count, average the two middle values.
    last = windows.shape[-1] - 1
    lo = np.clip((n_valid - 1) // 2, 0, last)
    hi = np.clip(n_valid // 2, 0, last)

    a = np.take_along_axis(ordered, lo[..., None], -1)[..., 0]
    b = np.take_along_axis(ordered, hi[..., None], -1)[..., 0]

    med = np.where(n_valid > 0, 0.5 * (a + b), np.nan).astype(np.float32)
    return med, n_valid


def _gauss_ksize(sigma_px):
    """Kernel size covering +-3 sigma: 2 * ceil(3 sigma) + 1."""
    return 2 * int(np.ceil(3 * sigma_px)) + 1


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------

def raw(depth):
    return depth.copy()


# ---------------------------------------------------------------------------
# Median filters
# ---------------------------------------------------------------------------

def median(depth, k=5):
    """
    Plain k x k median (scipy, border = nearest).

    Missing pixels enter the window as 0 m. So the filter can fill a hole (when most of the
    window is valid) and can also delete a measured pixel (when most of the window is missing).
    """
    z = np.nan_to_num(depth, nan=0.0)
    out = ndimage.median_filter(z, size=k, mode="nearest")
    return _clean(out)


def iamedian(depth, k=5):
    """
    Invalid-aware median: the median of the valid pixels in the k x k window.

    Only measured pixels are replaced. Holes stay holes and no measured pixel is deleted.
    """
    ok = valid(depth)
    med, _ = _valid_window_median(depth, ok, k)

    out = depth.copy()
    out[ok] = med[ok]
    return _clean(out)


# ---------------------------------------------------------------------------
# Gaussian filters
# ---------------------------------------------------------------------------

def gaussian_cv2(depth, sigma_px=1.0):
    """
    OpenCV Gaussian blur, kernel 2 * ceil(3 sigma) + 1, border = reflect-101.

    Missing pixels enter as 0 m, so values next to a hole are pulled towards 0.
    Input holes are set back to missing afterwards.
    """
    z = np.nan_to_num(depth, nan=0.0).astype(np.float32)
    k = _gauss_ksize(sigma_px)

    out = cv2.GaussianBlur(z, (k, k), sigmaX=float(sigma_px), sigmaY=float(sigma_px))

    out[~valid(depth)] = np.nan
    return _clean(out)


def gaussian_masked(depth, sigma_px=1.0):
    """
    Normalised Gaussian: blur(depth * valid) / blur(valid).

    Each output is a weighted mean over valid neighbours only. Pixels outside the image count
    as missing. Holes stay holes.
    """
    ok = valid(depth)
    k = _gauss_ksize(sigma_px)
    z = np.where(ok, depth, 0).astype(np.float64)

    num = cv2.GaussianBlur(z, (k, k), float(sigma_px), borderType=cv2.BORDER_CONSTANT)
    den = cv2.GaussianBlur(ok.astype(np.float64), (k, k), float(sigma_px), borderType=cv2.BORDER_CONSTANT)

    out = np.full(depth.shape, np.nan, np.float32)
    out[ok] = (num[ok] / den[ok]).astype(np.float32)
    return _clean(out)


# ---------------------------------------------------------------------------
# Bilateral filters
# ---------------------------------------------------------------------------

def bilateral_cv2(depth, d=9, sigma_space_px=5.0, sigma_range_m=0.03):
    """
    OpenCV bilateral filter on float32 metres (circular window of diameter d, border = reflect-101).

    Missing pixels enter as 0 m. With a small sigma_range a 0 m neighbour gets almost no weight;
    with a large one (above ~0.3 m) it pulls valid pixels towards 0.
    Input holes are set back to missing afterwards (OpenCV would write ~0 m there).
    """
    z = np.nan_to_num(depth, nan=0.0).astype(np.float32)

    out = cv2.bilateralFilter(z, d=int(d), sigmaColor=float(sigma_range_m), sigmaSpace=float(sigma_space_px))

    out[~valid(depth)] = np.nan
    return _clean(out)


def bilateral_masked(depth, d=9, sigma_space_px=5.0, sigma_range_m=0.03):
    """
    Normalised bilateral filter that ignores missing pixels.

    weight(neighbour) = exp(-r^2 / 2 sigma_space^2) * exp(-dz^2 / 2 sigma_range^2)
    over valid neighbours only, in the same circular window as OpenCV (radius d // 2).
    Pixels outside the image count as missing. Holes stay holes.
    """
    ok = valid(depth)
    z = np.where(ok, depth, 0).astype(np.float64)
    H, W = z.shape

    r = int(d) // 2
    z_pad = np.pad(z, r)
    ok_pad = np.pad(ok, r)

    num = np.zeros_like(z)
    den = np.zeros_like(z)

    # One shifted copy of the image per window offset; accumulate weighted sums.
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue  # outside the circular window

            z_n = z_pad[r + dy:r + dy + H, r + dx:r + dx + W]
            ok_n = ok_pad[r + dy:r + dy + H, r + dx:r + dx + W]

            w_space = np.exp(-(dx * dx + dy * dy) / (2 * sigma_space_px ** 2))
            w_range = np.exp(-((z_n - z) ** 2) / (2 * sigma_range_m ** 2))
            w = w_space * w_range * ok_n

            num += w * z_n
            den += w

    out = np.full(z.shape, np.nan, np.float32)
    out[ok] = (num[ok] / den[ok]).astype(np.float32)
    return _clean(out)


# ---------------------------------------------------------------------------
# Hole filling
# ---------------------------------------------------------------------------

def hole_fill(depth, k=5, max_hole_area=1024, edge_guard_m=0.10, min_neighbors=3, max_iters=100):
    """
    Fill small interior holes from their boundary inward with the median of valid neighbours.

    - A hole is eligible only if its area <= max_hole_area and it does not touch the image border.
    - Each pass fills the hole pixels that have at least min_neighbors valid pixels in their
      k x k window; the next pass can then use those as neighbours.
    - Edge guard: if the valid neighbours of a pixel span more than edge_guard_m, the hole sits on
      a depth edge and that pixel is left empty (None switches the guard off).
    - Measured pixels are never changed.
    """
    ok = valid(depth)
    H, W = depth.shape

    # Which holes (connected missing regions) may be filled?
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats((~ok).astype(np.uint8), connectivity=8)

    eligible = np.zeros(n_labels, bool)
    for i in range(1, n_labels):
        x, y, w, h, area = stats[i]
        touches_border = x == 0 or y == 0 or x + w == W or y + h == H
        eligible[i] = area <= max_hole_area and not touches_border

    target = eligible[labels] & ~ok

    out = depth.copy()
    filled = ok.copy()
    r = k // 2

    for _ in range(max_iters):
        todo = target & ~filled
        if not todo.any():
            break

        med, n_valid = _valid_window_median(out, filled, k)
        candidates = todo & (n_valid >= min_neighbors)

        if edge_guard_m is not None:
            padded = np.pad(np.where(filled, out, np.nan), r, constant_values=np.nan)
            windows = sliding_window_view(padded, (k, k))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows
                spread = np.nanmax(windows, axis=(-1, -2)) - np.nanmin(windows, axis=(-1, -2))

            on_edge = candidates & (spread > edge_guard_m)
            target &= ~on_edge  # never retried
            candidates &= ~on_edge

        if not candidates.any():
            break

        out[candidates] = med[candidates]
        filled |= candidates

    return _clean(out)


def inpaint(depth, method="fmm", radius_px=5, max_hole_area=None):
    """
    Fill holes with OpenCV inpainting.

    method          "fmm" = Telea's Fast Marching Method, "ns" = Navier-Stokes
    radius_px       neighbourhood each filled pixel is computed from
    max_hole_area   only fill holes up to this many pixels (None = fill every hole, border holes too)

    FMM extrapolates from the depth gradient at the hole's edge and can overshoot (even below 0 m),
    so every filled value is clipped to the depth range of the measured pixels around its own hole.
    Measured pixels are never changed.
    """
    ok = valid(depth)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats((~ok).astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return _clean(depth.copy())

    # Which holes to fill
    area = stats[:, cv2.CC_STAT_AREA]
    eligible = np.ones(n_labels, bool) if max_hole_area is None else area <= max_hole_area
    eligible[0] = False                      # label 0 = measured pixels
    target = eligible[labels]

    z = np.nan_to_num(depth, nan=0.0).astype(np.float32)
    flag = cv2.INPAINT_TELEA if method == "fmm" else cv2.INPAINT_NS
    filled = cv2.inpaint(z, target.astype(np.uint8), float(radius_px), flag)

    # Depth range of the measured ring around each hole (ring pixel -> label of the hole next to it)
    ring = ndimage.grey_dilation(labels, size=(3, 3)) * ok
    ids = np.arange(n_labels)
    lo = np.asarray(ndimage.minimum(z, labels=ring, index=ids), np.float32)
    hi = np.asarray(ndimage.maximum(z, labels=ring, index=ids), np.float32)

    out = depth.copy()
    out[target] = np.clip(filled[target], lo[labels][target], hi[labels][target])
    return _clean(out)


# ---------------------------------------------------------------------------
# Initial pipeline: noisy -> denoised -> restored
# ---------------------------------------------------------------------------

FILLS = ("none", "region", "fmm", "ns")


def pipeline(depth, filter="iamedian", fill="fmm", **params):
    """
    Denoise, then fill the holes that remain.

        noisy depth --filter--> denoised depth --fill--> restored depth

    filter   any denoising filter in FILTERS ("iamedian", "gaussian_masked", "bilateral_cv2", ...) or "none"
    fill     "none", "region" (hole_fill), "fmm" or "ns" (inpaint)
    params   settings for the filter by their own names (k, sigma_px, d, ...);
             settings for the fill step with a "fill_" prefix (fill_radius_px, fill_max_hole_area, ...)

    The 0.4 m gate is applied before this function (run.py: "min_depth_m"; pipeline.py: --gate).
    """
    filter_params = {k: v for k, v in params.items() if not k.startswith("fill_")}
    fill_params = {k[len("fill_"):]: v for k, v in params.items() if k.startswith("fill_")}

    # Step 1: denoise
    denoised = depth if filter == "none" else FILTERS[filter](depth, **filter_params)

    # Step 2: fill the remaining holes
    if fill == "none":
        return denoised
    if fill == "region":
        return hole_fill(denoised, **fill_params)
    if fill in ("fmm", "ns"):
        return inpaint(denoised, method=fill, **fill_params)

    raise ValueError(f"unknown fill method {fill!r}; use one of {FILLS}")


# ---------------------------------------------------------------------------
# Registry and config names
# ---------------------------------------------------------------------------

FILTERS = {
    "raw": raw,
    "median": median,
    "iamedian": iamedian,
    "gaussian_cv2": gaussian_cv2,
    "gaussian_masked": gaussian_masked,
    "bilateral_cv2": bilateral_cv2,
    "bilateral_masked": bilateral_masked,
    "hole_fill": hole_fill,
    "inpaint": inpaint,
    "pipeline": pipeline,
}

# Short parameter names used in config ids.
_SHORT = {
    "k": "k",
    "d": "d",
    "sigma_space_px": "s",
    "sigma_range_m": "r",
    "sigma_px": "s",
    "max_hole_area": "a",
    "edge_guard_m": "g",
    "radius_px": "rad",
    "filter": "",            # pipeline: the filter name is written as is
    "fill": "fill",
    "fill_radius_px": "rad",
    "fill_max_hole_area": "fa",
    "fill_k": "fk",
    "fill_edge_guard_m": "fg",
}


def make_id(method, params, min_depth_m=None):
    """Config name, e.g. make_id("iamedian", {"k": 7}, 0.4) -> "iamedian_k7_gate0.4"."""
    parts = [method]

    for name, value in params.items():
        if value is None:
            text = "none"
        elif isinstance(value, str):
            text = value
        else:
            text = format(value, "g")
        parts.append(f"{_SHORT.get(name, name)}{text}")

    if min_depth_m:
        parts.append(f"gate{min_depth_m:g}")

    return "_".join(parts)
