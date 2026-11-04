# =====================================================================
# STEP 1: INTRA-FRAME PADDING & LARGE VOID GUARD (Fig. 4 Pre-step)
# =====================================================================
def intra_frame_padding(raw_depth, min_void_size=100, pad_ksize=5):
    """
    1. Identifies large out-of-range voids (which the paper states should NOT be filled).
    2. Isolates small holes (Z == 0 that are not in large voids).
    3. Pads small holes with the average of their valid spatial neighbors.
    """
    # Convert to float32 so math doesn't overflow 16-bit integers
    depth = raw_depth.astype(np.float32)
    valid_mask = (depth > 0)
    hole_mask = ~valid_mask

    # Use Connected Components to find massive unrecoverable voids (mirrors/out-of-range)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        hole_mask.astype(np.uint8) * 255, connectivity=8
    )
    large_void_mask = np.zeros_like(hole_mask, dtype=bool)
    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] >= min_void_size:
            large_void_mask[labels == idx] = True

    # Small holes are zero-depth pixels that are NOT part of a massive void
    small_hole_mask = hole_mask & (~large_void_mask)

    # Compute spatial average of ONLY valid neighboring pixels using boxFilter
    # Sum of valid depths in (pad_ksize x pad_ksize) divided by count of valid pixels
    valid_float = valid_mask.astype(np.float32)
    sum_valid = cv2.boxFilter(depth, ddepth=cv2.CV_32F, ksize=(pad_ksize, pad_ksize), normalize=False)
    count_valid = cv2.boxFilter(valid_float, ddepth=cv2.CV_32F, ksize=(pad_ksize, pad_ksize), normalize=False)

    # Avoid division by zero if a hole is larger than pad_ksize
    spatial_avg = np.where(count_valid > 0, sum_valid / np.maximum(count_valid, 1.0), 0.0)

    # For slightly wider small holes, do a second pass so the interior gets padded too
    padded_depth = depth.copy()
    padded_depth[small_hole_mask] = spatial_avg[small_hole_mask]
    
    still_zero = small_hole_mask & (padded_depth == 0)
    if np.any(still_zero):
        valid_2 = (padded_depth > 0).astype(np.float32)
        sum_2 = cv2.boxFilter(padded_depth, ddepth=cv2.CV_32F, ksize=(pad_ksize + 4, pad_ksize + 4), normalize=False)
        cnt_2 = cv2.boxFilter(valid_2, ddepth=cv2.CV_32F, ksize=(pad_ksize + 4, pad_ksize + 4), normalize=False)
        padded_depth[still_zero] = np.where(cnt_2[still_zero] > 0, sum_2[still_zero] / np.maximum(cnt_2[still_zero], 1.0), 0.0)
    return padded_depth, valid_mask, small_hole_mask, large_void_mask


# =====================================================================
# STEP 2: PER-PIXEL MOTION VECTORS & MINIMUM MSE (Section III-A & Fig. 5)
# =====================================================================
def compute_motion_vectors(padded_frames, ref_valid_mask, k=5, search_radius=6):
    """
    Finds the optimal per-pixel shift (dy, dx) and minimum MSE between the
    available (valid) pixels in the reference frame and the padded search frames.
    
    Input Arguments:
        padded_frames: Frames which have had small holes padded using the average of their neighbours
        ref_valid_mask: A mask of the image, indicating which pixels had valid depth information
        k: Width of block in the spatial domain, centred around an invalid pixel value


    Returns:
        motion_vectors: array of shape (H, W, n_frames, 2) holding [dy, dx]
        min_mse_map:    array of shape (H, W, n_frames) holding the best MSE(i)
        matched_depths: array of shape (H, W, n_frames) holding I_{t+i}(y+dy, x+dx)
    """
    n_frames = len(padded_frames)
    ref_idx = n_frames // 2
    ref_frame = padded_frames[ref_idx]
    H, W = ref_frame.shape

    motion_vectors = np.zeros((H, W, n_frames, 2), dtype=np.int32)
    min_mse_map = np.full((H, W, n_frames), np.inf, dtype=np.float32)
    matched_depths = np.zeros((H, W, n_frames), dtype=np.float32)

    # Reference frame, with respect to itself, has 0 MSE and 0 shift
    min_mse_map[:, :, ref_idx] = 0.0 
    matched_depths[:, :, ref_idx] = ref_frame 

    # Count how many ORIGINAL valid ("available") pixels exist in each (k x k) reference block
    ref_valid_float = ref_valid_mask.astype(np.float32)
    avail_count = cv2.boxFilter(ref_valid_float, ddepth=cv2.CV_32F, ksize=(k, k), normalize=false)
    avail_count = np.maximum(avail_count, 1)

    # Pad search frames so shifting by +/- search_radius never goes out of bounds
    R = search_radius
    for f_idx in range(n_frames):
        if f_idx == ref_idx:
            continue

        search_frame = padded_frames[f_idx]
        search_padded = np.pad(search_frame, R, mode='edge')

        # Slide the search frame across all candidate shifts (dy, dx) in [-R, R]
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                # Shifted candidate image I_{t+i}(y + dy, x + dx)
                cand_shifted = search_padded[R + dy : R + dy + H, R + dx : R + dx + W]

                # Squared error ONLY at originally available pixels in the reference frame
                # (excludes the center hole and any unpadded holes in the reference block!)
                sq_diff = ((ref_frame - cand_shifted) ** 2) * ref_valid_float

                # Sum squared errors inside every (k x k) block simultaneously
                block_sse = cv2.boxFilter(sq_diff, ddepth=cv2.CV_32F, ksize=(k, k), normalize=False)
                block_mse = block_sse / avail_count

                # Where this shift (dy, dx) beats the previous best MSE, update!
                better = block_mse < min_mse_map[:, :, f_idx]
                min_mse_map[:, :, f_idx][better] = block_mse[better]
                motion_vectors[:, :, f_idx, 0][better] = dy
                motion_vectors[:, :, f_idx, 1][better] = dx
                matched_depths[:, :, f_idx][better] = cand_shifted[better]

    return motion_vectors, min_mse_map, matched_depths


# =====================================================================
# STEP 3 & 4: FULL PAPER PIPELINE (HOLE PREDICTION + 2D+TIME DNBL)
# =====================================================================
def paper_pipeline(
    raw_frames,
    r=1,
    m=2,
    k=5,
    search_radius=6,
    sigma_mse=500.0,
    sigma_s=2.0,
    sigma_r=2.0,
    k_noise=0.002,
    depth_scale=5000.0
):
    """
    Full implementation of 'Kinect-Like Depth Denoising' (Fu et al., 2012).
    
    Parameters:
        raw_frames:    List of (2 * max(r, m) + 1) uint16 depth frames, centered at frame t.
        r:             Temporal radius for Spatiotemporal DNBL (uses 2r + 1 frames).
        m:             Temporal radius for Inter-Frame Hole Prediction (uses 2m frames, skipping i=0).
        k:             Block/window size (k x k) for template matching and DNBL filtering.
        search_radius: Max pixel shift searched per frame for motion vectors.
        sigma_mse:     Gaussian parameter sigma for MSE weighting in Eq. (8)-(9).
        sigma_s:       Spatial Gaussian parameter for DNBL in Eq. (10) & (14).
        sigma_r:       Range multiplier for adaptive DNBL in Eq. (12) & (14).
        k_noise:       Sensor noise constant for Theta(I(p)) (default ~2mm std at 1m).
        depth_scale:   Scale factor converting uint16 units to meters (5000.0 for TUM/ICL-NUIM).
    """
    n_frames = len(raw_frames)
    ref_idx = n_frames // 2

    # -----------------------------------------------------------------
    # STAGE 0: Intra-Frame Padding on all loaded frames
    # -----------------------------------------------------------------
    padded_frames = []
    valid_masks = []
    small_hole_masks = []
    large_void_masks = []

    for f in raw_frames:
        pad_f, val_m, small_m, large_m = intra_frame_padding(f, min_void_size=100, pad_ksize=k)
        padded_frames.append(pad_f)
        valid_masks.append(val_m)
        small_hole_masks.append(small_m)
        large_void_masks.append(large_m)

    ref_padded = padded_frames[ref_idx]
    ref_valid = valid_masks[ref_idx]
    ref_small_holes = small_hole_masks[ref_idx]
    ref_large_voids = large_void_masks[ref_idx]
    H, W = ref_padded.shape

    # -----------------------------------------------------------------
    # STAGE 1: Compute Motion Vectors & Minimum MSE across frames
    # -----------------------------------------------------------------
    motion_vectors, min_mse_map, matched_depths = compute_motion_vectors(
        padded_frames, ref_valid, k=k, search_radius=search_radius
    )

    # -----------------------------------------------------------------
    # STAGE 2: Inter-Frame Hole Value Prediction (Equations 7, 8, 9)
    # -----------------------------------------------------------------
    hole_num_sum = np.zeros((H, W), dtype=np.float32)
    hole_weight_sum = np.zeros((H, W), dtype=np.float32)

    lower_m = ref_idx - m
    upper_m = ref_idx + m + 1

    for f_idx in range(lower_m, upper_m):
        # Skip i = 0 (ref_idx) so we don't double-count local intra-frame padding!
        if f_idx == ref_idx:
            continue

        mse_i = min_mse_map[:, :, f_idx]
        matched_val = matched_depths[:, :, f_idx]

        # Equation (8) & (9): Gaussian weight from MSE (front constant cancels out!)
        omega_i = np.exp(-(mse_i ** 2) / (2.0 * (sigma_mse ** 2)))
        
        # Only accumulate if the matched pixel in frame t+i has a positive depth
        valid_match = (matched_val > 0).astype(np.float32)
        w = omega_i * valid_match

        hole_num_sum += w * matched_val
        hole_weight_sum += w

    # Start with the original valid pixels of the reference frame
    hole_filled_ref = ref_padded.copy()
    
    # Where inter-frame weights are valid, use the weighted temporal prediction;
    # otherwise fall back safely to the intra-frame spatial average!
    inter_frame_pred = np.where(
        hole_weight_sum > 1e-6,
        hole_num_sum / np.maximum(hole_weight_sum, 1e-6),
        ref_padded
    )
    hole_filled_ref[ref_small_holes] = inter_frame_pred[ref_small_holes]
    hole_filled_ref[ref_large_voids] = 0.0

    # Update the center frame in our buffer with the newly predicted holes
    padded_frames[ref_idx] = hole_filled_ref

    # -----------------------------------------------------------------
    # STAGE 3: 2D+Time Spatiotemporal DNBL Filter (Equations 12 & 14)
    # -----------------------------------------------------------------
    # Convert center frame I(p) to meters so Theta(I(p)) is easy to reason about
    I_p = hole_filled_ref / depth_scale
    valid_p = (I_p > 0)

    # Equation (5) & (12): Noise std scales with I(p)^2, so Variance Theta(I(p)) = (k_noise * I(p)^2)^2
    sigma_noise_p = np.maximum(k_noise * (I_p ** 2), 1e-4)
    theta_I_p = sigma_noise_p ** 2
    denom_range = 2.0 * (sigma_r ** 2) * theta_I_p

    dnbl_num = np.zeros((H, W), dtype=np.float32)
    W_p = np.zeros((H, W), dtype=np.float32)

    half_k = k // 2
    lower_r = ref_idx - r
    upper_r = ref_idx + r + 1

    # Grid of base pixel coordinates (y, x)
    grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # Outer sum of Eq. (14): Loop over temporal window j in [-r, r]
    for f_idx in range(lower_r, upper_r):
        frame_meters = padded_frames[f_idx] / depth_scale

        # Motion trajectory center (y + dy_j, x + dx_j) for frame f_idx
        dy_j = motion_vectors[:, :, f_idx, 0]
        dx_j = motion_vectors[:, :, f_idx, 1]

        # Inner sum of Eq. (14): Loop over q_j in local (k x k) window S_{t+j}(p)
        for wy in range(-half_k, half_k + 1):
            for wx in range(-half_k, half_k + 1):
                # Spatial Gaussian G_{sigma_s}(||p - q_j||)
                w_spatial = np.exp(-(wy**2 + wx**2) / (2.0 * (sigma_s ** 2)))

                # Coordinates of neighbor q_j around the motion-compensated trajectory center
                qy = np.clip(grid_y + dy_j + wy, 0, H - 1)
                qx = np.clip(grid_x + dx_j + wx, 0, W - 1)

                I_qj = frame_meters[qy, qx]
                valid_qj = (I_qj > 0).astype(np.float32)

                # Equation (12): Divisive Normalized Range Gaussian G'_{sigma_r}(||I(p) - I(q_j)||)
                diff_sq = (I_p - I_qj) ** 2
                w_range = np.exp(-diff_sq / denom_range)

                # Total weight for neighbor q_j
                weight = w_spatial * w_range * valid_qj

                dnbl_num += weight * I_qj
                W_p += weight

    # Final normalization by W_p (and keep large out-of-range voids strictly at 0)
    denoised_meters = np.where((W_p > 0) & valid_p & (~ref_large_voids), dnbl_num / np.maximum(W_p, 1e-6), 0.0)
    
    return (denoised_meters * depth_scale).astype(np.uint16)