#!/bin/sh
#
# Run every tuning sweep (ICL tuning frames 0:1509:50), one after another.
#
#   sh denoise/run_tuning.sh > outputs/tuning_log.txt 2>&1
#
# Runs that already have results are skipped by run.py (it never overwrites).
# The *_ext runs extend the bilateral grids, whose first minima sat on the grid edge.

cd "$(dirname "$0")/.." || exit 1

PY=.venv/Scripts/python.exe

for e in median iamedian hole_fill \
         gaussian_cv2 gaussian_masked \
         bilateral_cv2 bilateral_masked \
         bilateral_cv2_ext bilateral_masked_ext
do
    $PY denoise/run.py "experiments/tune_$e.json" --workers 4 || echo "FAILED or skipped: tune_$e"
done
