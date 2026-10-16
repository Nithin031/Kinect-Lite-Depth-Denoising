# Progress report: work done after midsem

*Updated 16-10-2026*

Sir, apologies for the late update. We had to work through a lot before finalising the denoising pipeline,
and some parts turned out trickier than expected. Here is our progress.

Depth cameras like the Kinect give a distance for every pixel, but the images are noisy: some distances are
wrong and some pixels have no value at all. Since we want to use these images for camera tracking (ICP), our
first task was to clean them.


## Temporal filtering: tried and dropped

We also tried temporal filtering, alone and combined with spatial filtering: combining several consecutive
frames, aligned using the camera's known poses, so that each pixel is averaged over time. It did not help.
Compared with spatial filtering alone, the results stayed about the same, and in some cases they were worse.
So we dropped temporal filtering and kept only the spatial pipeline.


## The spatial pipeline

The pipeline has three steps:

1. delete depth values closer than 0.4 m, which the sensor cannot measure;
2. denoise each pixel using its neighbours (a median filter);
3. fill the remaining holes from the surrounding depth (inpainting).

No machine learning is involved: these are classical image filters, and we only tuned a few settings (such as
the window size) by trying values and keeping the best.


## Results

To judge the filters fairly we used the simulated ICL-NUIM dataset, where the true depth of every pixel is
known. Analysing the data first gave us the most important insight: most of the error is not gentle noise but
sharp spikes at object edges. This explains the main result. The median filter, which ignores outliers,
clearly beat the Gaussian and bilateral filters, which smear spikes into their neighbours. Over all 1,509
frames it reduced the average depth error from 34.7 mm to 13.0 mm and improved every single frame. Hole
filling makes the image complete, but the filled values are estimates and less accurate.

On the real TUM dataset there is no true depth, so the results cannot be scored directly. Real data also has
far more holes (about 17 % of pixels), so we cannot yet say which filter works best in practice.


## Next steps

1. Implement ICP to estimate the camera's motion from the depth images.
2. Compare the paths estimated from raw and cleaned depth with TUM's true camera path. This will show whether
   denoising actually improves tracking on a real sensor.

The full method and results are in [`PIPELINE.md`](PIPELINE.md); the main figures are in
[`../RESULTS/`](../RESULTS/README.md).
