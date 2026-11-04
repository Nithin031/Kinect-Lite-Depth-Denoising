# Experiments

The hole-filling comparison, and a few files kept from the earlier study; the final code is in `../src/`.

| folder | what it is |
|---|---|
| `hole_filling/` | `compare_fills.py`: eight ways to fill the holes left after the 9x9 median (none, Telea, Navier-Stokes, nearest, region earlier, region median, background, temporal from the paper), scored by fill error (ICL against true depth; masked test on ICL and TUM) and by camera tracking with both trackers. Run on Kaggle (`../kaggle/notebook_fills/`). `results/`: `fills.txt` (all numbers), `fills.png` (summary chart), `fill_example.png` (one ICL hole, every fill). Result: no fill improves RGB-D odometry; Telea is among the least accurate; the temporal fill is the most accurate but about 8 s per frame. |
| `earlier_paper_implementations/` | our two first implementations of the paper (Fu et al., 2012). They disagreed on how the hole's own frame enters the hole prediction; `src/paper_pipeline.py` merges them and turns that choice into the `center` switch (`exclude` = `pipeline_paper_kt.py`, `nonlocal` = `pipeline_v2.py`). |
| `filter_study/` | the study that chose our method: how each filter behaves (`03`), the error of every filter on flat areas and at edges (`04`, `results_table.csv`), the filters in 3D (`05`), and the final pipeline on ICL and TUM frames (`09`, `10`). The median won because the noise is mostly spikes, which a median removes and an average smears. |

Settings were tuned on frames 0, 50, 100, ... and scored on frames 25, 75, 125, ..., so no result was tuned on the
frames it was scored on.
