# Results: start here

Depth images from a Kinect-style camera are noisy. This project tests simple spatial filters
(median, Gaussian, bilateral, hole filling) on them and measures how much each one helps.

**Main result.** On the ICL-NUIM test frames (simulated noise, true depth known), the two median filters
roughly halve the average depth error (26.2 mm to 12.8-14.8 mm) and remove about two-thirds of the error
at object edges. Gaussian and bilateral filters help less on flat areas and much less at edges.


## Look at these in order

| file | what it shows |
|---|---|
| `01_depth_image_to_point_cloud.png` | a colour image, its depth image, and the same pixels lifted into a 3D point cloud |
| `02_what_the_noise_looks_like_in_3d.png` | noisy vs true depth in 3D; from above, the bad points form lines from the camera to object edges |
| `03_how_each_filter_works.png` | a made-up row with an edge and two spikes, through each filter: median votes spikes out, Gaussian spreads them |
| `04_results_chart.png` | **the main result**: error on flat areas and at edges, for each filter |
| `results_table.csv` | the numbers behind the chart, plus run time per frame |
| `05_filters_in_3d_icl.png` | the same frame from above after each filter |
| `06_error_images_icl_frame775.png` | error images (darker = larger error) for each filter on one frame |
| `07_real_sensor_tum.png` | a real Asus Xtion frame (TUM): colour, depth with missing pixels, point cloud |
| `08_filters_in_3d_tum.png` | the filters on real data: Gaussian and bilateral create points between surfaces |
| `09_initial_pipeline_icl.png` | the initial pipeline on one ICL frame: noisy, denoised, restored, with error maps |
| `10_initial_pipeline_tum.png` | the same on a real TUM frame: what the filter changed and which pixels were filled |
| `3d_viewer_icl.html`, `3d_viewer_tum.html` | open in a browser, then rotate and zoom the point clouds; click a name to show / hide it (needs internet once, to load plotly) |


## The numbers (ICL, 30 test frames, 0.4 m gate)

| filter | all pixels (mm) | flat areas (mm) | object edges (mm) | frames worse than raw |
|---|---|---|---|---|
| raw input | 26.2 | 14.1 | 336 | - |
| median, k 11 | 14.8 | 10.8 | 115 | 0 |
| invalid-aware median, k 9 | 12.8 | 9.2 | 107 | 0 |
| Gaussian, OpenCV, sigma 0.5 px | 26.1 | 14.5 | 322 | 12 |
| Gaussian, masked, sigma 1.5 px | 23.7 | 12.5 | 289 | 0 |
| bilateral, OpenCV, sigma_r 0.3 m | 23.4 | 12.1 | 306 | 0 |
| bilateral, masked, sigma_r 1 m | 21.5 | 11.6 | 236 | 0 |

Error = mean absolute difference from the true depth. Lower is better.


## Read with these limits
- ICL's noise is simulated; its edge spikes look like a side effect of the simulation, so the median's large
  edge advantage may be smaller on a real camera.
- TUM (real Asus Xtion) has no true depth: there we can show what a filter changes, not whether it is right.
- Settings were tuned on other frames (0, 50, 100 ...) than the ones scored here (25, 75, 125 ...).


## More detail
- Full write-ups in `docs/reports/` (PDF and Word): theory to results, the results with all charts, and a project guide.
- Every run's raw numbers: `outputs/<run name>/summary.csv`
- Rebuild this folder: `.venv\Scripts\python.exe denoise\make_results.py`
