# References

Bibliographic details were checked against publisher, DOI or repository records on 2026-10-15.
No published numbers are compared with our measurements: datasets, frames, masks and metrics differ.


## Datasets and their noise

| # | Reference | Used for |
|---|---|---|
| A1 | A. Handa, T. Whelan, J. McDonald, A. J. Davison, "A Benchmark for RGB-D Visual Odometry, 3D Reconstruction and SLAM," IEEE ICRA 2014, pp. 1524-1531. DOI [10.1109/ICRA.2014.6907054](https://doi.org/10.1109/ICRA.2014.6907054). | ICL-NUIM `lr_kt0` and its depth noise model (Section V-A, Eq. 3): random pixel shifts with bilinear interpolation (sigma 0.5 px), Gaussian disparity noise (sigma 1/6), rounding of the disparity 35130/Z. Intrinsics (fy listed as -480) on the [codes page](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html). Also defines the trajectory error (ATE) used for ICP evaluation. |
| A2 | J. Sturm, N. Engelhard, F. Endres, W. Burgard, D. Cremers, "A Benchmark for the Evaluation of RGB-D SLAM Systems," IEEE/RSJ IROS 2012. DOI [10.1109/IROS.2012.6385773](https://doi.org/10.1109/IROS.2012.6385773) | TUM fr3 long_office_household, the depth PNG format (value / 5000 = metres), the motion-capture camera poses, and the ATE / RPE trajectory metrics. |
| A3 | J. T. Barron, J. Malik, "Intrinsic Scene Properties from a Single RGB-D Image," IEEE CVPR 2013, pp. 17-24. DOI [10.1109/CVPR.2013.10](https://doi.org/10.1109/CVPR.2013.10) | Cited by A1 as the origin of its shift-and-quantise Kinect noise simulation. |


## Sensor noise

| # | Reference | Used for |
|---|---|---|
| B1 | K. Khoshelham, S. Oude Elberink, "Accuracy and Resolution of Kinect Depth Data for Indoor Mapping Applications," *Sensors* 12(2):1437-1454, 2012. DOI [10.3390/s120201437](https://doi.org/10.3390/s120201437) | Depth from disparity: random error and depth step grow roughly with the square of the distance. |
| B2 | C. V. Nguyen, S. Izadi, D. Lovell, "Modeling Kinect Sensor Noise for Improved 3D Reconstruction and Tracking," 3DIMPVT 2012, pp. 524-530. [IEEE CS listing](https://computer.org/csdl/proceedings-article/3dimpvt/2012/4873a524/12OmNAlvHUP) (DOI not confirmed) | Axial and lateral noise vs distance and surface angle; why errors concentrate at depth edges. |


## Filters

| # | Reference | Used for |
|---|---|---|
| C1 | C. Tomasi, R. Manduchi, "Bilateral Filtering for Gray and Color Images," ICCV 1998, pp. 839-846. DOI [10.1109/ICCV.1998.710815](https://doi.org/10.1109/ICCV.1998.710815) | Definition of the bilateral filter (spatial Gaussian x range Gaussian). The paper does not define handling of missing pixels; our masked version leaves them out. |
| C2 | J. Kopf, M. F. Cohen, D. Lischinski, M. Uyttendaele, "Joint Bilateral Upsampling," ACM TOG 26(3), 2007. DOI [10.1145/1276377.1276497](https://doi.org/10.1145/1276377.1276497) | RGB-guided bilateral filtering of depth. Not implemented. |
| C3 | K. He, J. Sun, X. Tang, "Guided Image Filtering," IEEE TPAMI 35(6):1397-1409, 2013. DOI [10.1109/TPAMI.2012.213](https://doi.org/10.1109/TPAMI.2012.213) | Named in the project proposal. Not implemented. |

Median, invalid-aware median, Gaussian and hole filling are standard constructions defined in `denoise/filters.py`.


## Next phase (registration)

| # | Reference | Relevance |
|---|---|---|
| D1 | R. A. Newcombe, S. Izadi, O. Hilliges, D. Molyneaux, D. Kim, A. J. Davison, P. Kohli, J. Shotton, S. Hodges, A. Fitzgibbon, "KinectFusion: Real-time dense surface mapping and tracking," IEEE ISMAR 2011, pp. 127-136. DOI [10.1109/ISMAR.2011.6092378](https://doi.org/10.1109/ISMAR.2011.6092378) | Bilateral-filters raw depth before ICP tracking (point-to-plane ICP, projective data association). The reference design for using a denoised depth map as ICP input. |
