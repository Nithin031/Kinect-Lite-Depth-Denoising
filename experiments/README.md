# experiments/

Each `.json` file here is a recipe: which filters to run, with which settings, on which frames.
One command runs a recipe and writes its results to `outputs/<name>/`:

```
.venv\Scripts\python.exe denoise\run.py experiments\<file>.json
```


## What a recipe looks like

```json
{
  "name": "tune_gaussian_masked",
  "dataset": "icl",
  "frames": "0:1509:50",
  "configs": [
    {"method": "gaussian_masked",
     "grid": {"sigma_px": [0.5, 0.75, 1, 1.5, 2, 3, 5]},
     "min_depth_m": [null, 0.4]}
  ]
}
```

| key | meaning |
|---|---|
| `name` | output folder: `outputs/<name>/` |
| `dataset` | `icl` (true depth known, gets error scores) or `tum` (real Asus Xtion, change only) |
| `frames` | `start:stop:step`; `0:1509:50` = frames 0, 50, 100 ... 1500 |
| `method` | a filter from `denoise/filters.py`; `pipeline` chains a filter and a fill (`filter`, `fill`, fill settings prefixed `fill_`) |
| `params` | fixed settings |
| `grid` | every listed value is tried; two settings with 3 values each = 9 runs |
| `min_depth_m` | `null` = no gate, `0.4` = depth below 0.4 m treated as missing; a list runs both |

The raw input is always added as the reference, so the recipe above runs 7 x 2 + 1 = 15 configs.


## The recipes, in the order they were run

| file | what it does | frames |
|---|---|---|
| `tune_median.json` | plain median, k = 3, 5, 7, 9, 11 | tuning (0:1509:50) |
| `tune_iamedian.json` | invalid-aware median, k = 3 to 13 | tuning |
| `tune_gaussian_cv2.json` | OpenCV Gaussian, sigma 0.5 to 5 px | tuning |
| `tune_gaussian_masked.json` | masked Gaussian, sigma 0.5 to 5 px | tuning |
| `tune_bilateral_cv2.json` | OpenCV bilateral: d 5/9/15 x sigma_space 2/5/10 x sigma_range 0.005-1 m (72 settings) | tuning |
| `tune_bilateral_masked.json` | masked bilateral, same 72 settings | tuning |
| `tune_bilateral_cv2_ext.json`, `tune_bilateral_masked_ext.json` | larger windows (d up to 31) and sigma_range (up to 10 m), because the first grid's best results sat on its edge | tuning |
| `tune_hole_fill.json` | hole filling: max hole area 64-4096 px x edge guard none/0.05/0.1/0.2 m | tuning |
| `eval_selected.json` | the lowest-error setting per filter family on the tuning frames, plus the original settings | test |
| `tum_selected.json` | the same settings on real Asus Xtion data | TUM 25:2509:50 |
| `pipeline_icl.json`, `pipeline_tum.json` | the initial pipeline: a denoising filter, then hole filling (none, region, FMM, Navier-Stokes) | test, TUM |

`eval_selected.json` and `tum_selected.json` are written by `denoise/select_configs.py`, not by hand.


## What a run writes to `outputs/<name>/`

| file | contents |
|---|---|
| `summary.csv` | one row per setting: read this first |
| `per_frame.csv` | one row per frame and setting |
| `experiment_resolved.json` | the exact list of settings and the command used |
| `environment.json` | library versions |
| `frames/<i>/*.npy` | filtered depth images, only with `--save-frames` |

A finished run is never overwritten (use `--name` for a new folder), so every number can be traced to a file.
