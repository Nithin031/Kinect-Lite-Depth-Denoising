# Running on Kaggle

Two notebooks run the slow parts on Kaggle (private, 4 CPU cores):

| notebook | runs | time | results go to |
|---|---|---|---|
| `notebook/` (kinect-lite-tum-compare) | `src/tum_compare.py`: all methods, both trackers, on TUM | ~45 min | `src/results/` |
| `notebook_fills/` (kinect-lite-hole-fills-8) | `experiments/hole_filling/compare_fills.py`: 8 hole-filling methods on ICL and TUM | see log | `experiments/hole_filling/results/` |

Both need three private datasets. Run from the project folder in PowerShell, after `kaggle auth login`.

```
# 1. code: the pipeline and the experiment, next to their Kaggle description
copy src\*.py kaggle\code\
copy experiments\hole_filling\compare_fills.py kaggle\code\
kaggle datasets create -p kaggle\code                    # later: kaggle datasets version -p kaggle\code -m "update"

# 2. TUM fr3 long_office_household (1.5 GB); Kaggle needs the description inside the folder
copy kaggle\tum-dataset-metadata.json "RGB TUM\rgbd_dataset_freiburg3_long_office_household\dataset-metadata.json"
kaggle datasets create -p "RGB TUM\rgbd_dataset_freiburg3_long_office_household" --dir-mode zip

# 3. ICL-NUIM kt0, frames 0-300 only (250 MB): copy lr_kt0_noisy\depth, lr_kt0_noisy\rgb, lr_kt0_noisefree\depth
#    (files 0.png .. 300.png) and gt\livingRoom0.gt.freiburg into one folder, add kaggle\icl-dataset-metadata.json
#    as dataset-metadata.json, then
kaggle datasets create -p <that folder> --dir-mode zip

# 4. wait until all say "ready"
kaggle datasets status wizard1803/kinect-lite-code
kaggle datasets status wizard1803/tum-fr3-long-office-household
kaggle datasets status wizard1803/icl-nuim-kt0-0-300

# 5. start a run, check it ("COMPLETE" when done), download the results
kaggle kernels push -p kaggle\notebook_fills
kaggle kernels status wizard1803/kinect-lite-hole-fills-8
kaggle kernels output wizard1803/kinect-lite-hole-fills-8 -p kaggle\output_fills
```

Then copy the downloaded `results/*` to the folder in the table above.

The notebooks install Open3D (Kaggle allows internet only on phone-verified accounts), unpack the zipped folders
into the layout `src/data.py` expects, and point `KINECT_DATA` (or `TUM_DIR`) at it.

To redo only the tracking of the TUM comparison locally: download the whole output, copy
`code\cache\tum_compare\frames` to `cache\tum_compare\` and run `src\tum_compare.py`; it reuses the restored frames.
