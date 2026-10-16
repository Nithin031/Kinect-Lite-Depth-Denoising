r"""
Pick one setting per filter family from the tuning runs and write the evaluation experiments.

Usage
-----
    .venv\Scripts\python.exe denoise\select_configs.py

Selection rule (fixed before the evaluation frames are scored)
--------------------------------------------------------------
For every method and gate (none / 0.4 m): the config with the lowest all_mae_mm on the tuning
frames (0:1509:50, matched mask of that family's tuning run).

- This picks a setting within each family. It does not rank the families against each other.
- A family may have several tuning runs (an extension grid when a minimum sat on the grid edge).
  They are pooled only if their raw rows have the same all_mae_mm, i.e. the same matched mask.
- hole_fill is not selected this way: it changes only holes, which the matched mask excludes, so
  all_mae_mm is the same for every hole_fill config. Its trade-off (more fills vs more fills > 5 cm
  off) is shown in figures/tuning_hole_fill.png; only the original default is carried forward.
- The original default configs are always added, so tuned settings can be compared with them.
  Gaussian was not in the original run, so it has no default.

Writes
------
    experiments/eval_selected.json    ICL evaluation frames 25:1509:50
    experiments/tum_selected.json     TUM frames 25:2509:50
    outputs/tuning_selection.csv      the chosen rows with their tuning-frame numbers
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D


# Family -> tuning runs (folders in outputs/)
FAMILIES = {
    "median": ["tune_median"],
    "iamedian": ["tune_iamedian"],
    "gaussian_cv2": ["tune_gaussian_cv2"],
    "gaussian_masked": ["tune_gaussian_masked"],
    "bilateral_cv2": ["tune_bilateral_cv2", "tune_bilateral_cv2_ext"],
    "bilateral_masked": ["tune_bilateral_masked", "tune_bilateral_masked_ext"],
}

# Settings of the original run, always included
DEFAULTS = [
    {"method": "median", "params": {"k": 5}},
    {"method": "iamedian", "params": {"k": 5}},
    {"method": "bilateral_cv2", "params": {"d": 9, "sigma_space_px": 5, "sigma_range_m": 0.03}},
    {"method": "bilateral_masked", "params": {"d": 9, "sigma_space_px": 5, "sigma_range_m": 0.03}},
    {"method": "hole_fill", "params": {"k": 5, "max_hole_area": 1024, "edge_guard_m": 0.1}},
]

OUTPUT_EXPERIMENTS = [
    ("eval_selected", "icl", "25:1509:50"),
    ("tum_selected", "tum", "25:2509:50"),
]


def load_family(fam, runs):
    """Pooled summary rows of one family + {config_id: resolved config}."""
    parts = []
    resolved = {}

    for name in runs:
        run = os.path.join(D.OUTPUTS, name)

        p = pd.read_csv(os.path.join(run, "summary.csv"))
        p["run"] = name
        parts.append(p)

        with open(os.path.join(run, "experiment_resolved.json")) as f:
            for c in json.load(f)["configs"]:
                resolved[c["id"]] = c

    # Pool only runs scored on the same pixels
    raws = [p.loc[p.config == "raw", "all_mae_mm"].iloc[0] for p in parts]
    if max(raws) - min(raws) > 1e-9:
        sys.exit(f"{fam}: tuning runs have different matched masks (raw all_mae {raws}); not pooled")

    s = pd.concat(parts).drop_duplicates("config")
    s = s[s.method == fam]
    return s, resolved


def main():
    chosen = []
    configs = []

    for fam, runs in FAMILIES.items():
        s, resolved = load_family(fam, runs)

        for gate, group in s.groupby("min_depth_m"):
            row = group.loc[group.all_mae_mm.idxmin()]
            c = resolved[row.config]

            chosen.append({
                "family": fam,
                "gate_m": gate,
                "config": row.config,
                "run": row.run,
                "n_configs_in_sweep": len(group),
                "tune_all_mae_mm": row.all_mae_mm,
                "tune_flat_mae_mm": row.flat_mae_mm,
                "tune_band2_mae_mm": row.band2_mae_mm,
                "tune_own_mae_mm": row.own_mae_mm_framemean,
                "tune_coverage_pct": row.coverage_pct_mean,
                "tune_frames_worse_than_raw": row.frames_worse_than_raw,
                "tune_frames": row.frames,
            })
            configs.append({"method": fam, "params": c["params"], "min_depth_m": c["min_depth_m"]})

    for d in DEFAULTS:
        configs.append({**d, "min_depth_m": [None, 0.4]})

    pd.DataFrame(chosen).to_csv(os.path.join(D.OUTPUTS, "tuning_selection.csv"), index=False)

    # Experiment files (never overwritten)
    doc = ("Written by denoise/select_configs.py: per method and gate, the lowest all_mae_mm "
           "on the tuning frames, plus the original defaults.")

    for name, dataset, frames in OUTPUT_EXPERIMENTS:
        path = os.path.join(D.ROOT, "experiments", f"{name}.json")
        if os.path.exists(path):
            sys.exit(f"{path} exists; not overwritten")

        exp = {"name": name, "_doc": doc, "dataset": dataset, "frames": frames, "configs": configs}
        with open(path, "w") as f:
            json.dump(exp, f, indent=1)

    print(pd.DataFrame(chosen).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
