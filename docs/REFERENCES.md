# References

## Datasets

- A. Handa, T. Whelan, J. McDonald, A. J. Davison, "A Benchmark for RGB-D Visual Odometry, 3D Reconstruction
  and SLAM," IEEE ICRA 2014, pp. 1524-1531. DOI [10.1109/ICRA.2014.6907054](https://doi.org/10.1109/ICRA.2014.6907054).
  The ICL-NUIM dataset and its depth noise model (random pixel shifts, Gaussian noise on the disparity,
  rounding of the disparity).
- J. Sturm, N. Engelhard, F. Endres, W. Burgard, D. Cremers, "A Benchmark for the Evaluation of RGB-D SLAM
  Systems," IEEE/RSJ IROS 2012. DOI [10.1109/IROS.2012.6385773](https://doi.org/10.1109/IROS.2012.6385773).
  The TUM RGB-D dataset, its depth format and its trajectory error metrics.

## Sensor noise

- K. Khoshelham, S. Oude Elberink, "Accuracy and Resolution of Kinect Depth Data for Indoor Mapping
  Applications," *Sensors* 12(2):1437-1454, 2012. DOI [10.3390/s120201437](https://doi.org/10.3390/s120201437).
  Depth error grows roughly with the square of the distance.

## Filters

- C. Tomasi, R. Manduchi, "Bilateral Filtering for Gray and Color Images," ICCV 1998, pp. 839-846.
  DOI [10.1109/ICCV.1998.710815](https://doi.org/10.1109/ICCV.1998.710815).
- A. Telea, "An Image Inpainting Technique Based on the Fast Marching Method," Journal of Graphics Tools 9(1),
  2004. The FMM inpainting used for hole filling (as implemented in OpenCV).
- M. Bertalmio, A. L. Bertozzi, G. Sapiro, "Navier-Stokes, Fluid Dynamics, and Image and Video Inpainting,"
  IEEE CVPR 2001. The Navier-Stokes inpainting option (as implemented in OpenCV).

## Next phase

- R. A. Newcombe et al., "KinectFusion: Real-time dense surface mapping and tracking," IEEE ISMAR 2011,
  pp. 127-136. DOI [10.1109/ISMAR.2011.6092378](https://doi.org/10.1109/ISMAR.2011.6092378).
  Filters raw depth before ICP tracking: the reference design for this project's next step.
