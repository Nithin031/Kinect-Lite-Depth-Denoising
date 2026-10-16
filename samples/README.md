# Sample frames

A few real frames so the pipeline can be tried without downloading the full datasets (about 5 MB).

| folder / file | what it is |
|---|---|
| `icl/noisy/*.png` | 5 ICL-NUIM depth frames with simulated sensor noise (frames 100, 400, 775, 1000, 1300) |
| `icl/true/*.png` | the noise-free (true) depth of the same 5 frames |
| `tum/depth/*.png` | 20 real depth frames from TUM fr3 long_office_household (frames 1100-1290, every 10th) |
| `tum_depth_video.avi` | the same 20 TUM frames as an 8-bit depth video (grey 0 = missing, 255 = 5 m), 3 fps |

All PNGs are 16-bit: value / 5000 = metres, 0 = no measurement.

Try:

```
python denoise/process.py --input samples/tum/depth
python denoise/process.py --input samples/icl/noisy/775.png --camera icl
python denoise/process.py --input samples/tum_depth_video.avi --video-max-depth 5
```

## Credits

- **ICL-NUIM**: A. Handa, T. Whelan, J. McDonald, A. J. Davison, "A Benchmark for RGB-D Visual Odometry,
  3D Reconstruction and SLAM", ICRA 2014. https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html
  (Creative Commons Attribution 3.0).
- **TUM RGB-D**: J. Sturm, N. Engelhard, F. Endres, W. Burgard, D. Cremers, "A Benchmark for the Evaluation of
  RGB-D SLAM Systems", IROS 2012. https://cvg.cit.tum.de/data/datasets/rgbd-dataset
  (Creative Commons Attribution 4.0).
