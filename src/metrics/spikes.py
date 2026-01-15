import numpy as np
import pandas as pd

def detect_loop_spikes(values, noise, smooth=None, window=40, amp_threshold=5e-10, baseline_window=3, baseline_tolerance=2.0):
    """
    Heuristic: "Return spikes".
    Checks if signal deviates significantly and then returns to the same baseline.
    
    Refinements:
    - Baseline Check: Checks start/end segments against the smooth trend.
    - Oscillation: Allows max 1 oscillation (M-shape), but rejects >1 (messy).
    """
    n = len(values)
    mask = np.zeros(n, dtype=bool)
    
    # Convert to numpy if series
    if isinstance(values, pd.Series): values = values.values
    if isinstance(noise, pd.Series): noise = noise.values
    if isinstance(smooth, pd.Series): smooth = smooth.values
    
    # If smooth not provided, estimate simple median trend
    if smooth is None:
        smooth = pd.Series(values).rolling(window=window, center=True, min_periods=1).median().values
        
    if n < window or window < 3:
        return mask

    # Fallback epsilon for noise
    eps = np.nanmedian(noise)
    if not np.isfinite(eps) or eps == 0:
        eps = np.nanmedian(np.abs(values - np.nanmedian(values))) * 0.1 + 1e-12

    bw = max(1, int(baseline_window))

    for start in range(0, n - window + 1):
        end = start + window
        
        # --- 1. Baseline Stability Check ---
        # Check if 'bw' points at start and end are close to the smooth trend
        # We use the local noise level as the scale factor.
        local_noise = np.nanmedian(noise[start:end])
        if not np.isfinite(local_noise) or local_noise == 0: local_noise = eps
            
        # Error = Mean Absolute Deviation from Trend
        start_seg_err = np.mean(np.abs(values[start : start + bw] - smooth[start : start + bw]))
        end_seg_err = np.mean(np.abs(values[end - bw : end] - smooth[end - bw : end]))
        
        threshold = baseline_tolerance * local_noise
        
        if start_seg_err < threshold and end_seg_err < threshold:
            
            # --- 2. Excursion Analysis ---
            seg = values[start:end]
            seg_smooth = smooth[start:end]
            
            # Identify points significantly off-trend
            # We use 2*noise as the definition of "off trend" inside the loop too
            is_off = np.abs(seg - seg_smooth) > 2 * local_noise
            off_idx = np.where(is_off)[0]
            
            if len(off_idx) == 0: continue
            
            # --- 3. Oscillation Check ---
            # Count how many times we return to baseline (gaps in off_idx)
            # Gap = index jump > 1
            num_gaps = np.sum(np.diff(off_idx) > 1)
            
            # User Req: "1 oscillation max" -> Allow 0 or 1 gaps (single bump or M-shape)
            if num_gaps > 1:
                continue

            # --- 4. Amplitude Check (No Direction Constraint) ---
            # User Req: "remove the need to be one sided... accept M but also up and down"
            # We just ensure the deviation is significant.
            
            start_val = values[start]
            excursion = seg[off_idx] - start_val
            deviation = np.max(np.abs(excursion))
            
            if deviation >= max(5 * local_noise, amp_threshold):
                mask[start + off_idx[0] : start + off_idx[-1] + 1] = True
                    
    return mask

def detect_residual_spikes(values, smooth, sigma_factor=4.0, min_threshold=1e-9):
    """
    Hampel/Z-score style detection on residuals.
    """
    if isinstance(values, pd.Series): values = values.values
    if isinstance(smooth, pd.Series): smooth = smooth.values
        
    residuals = values - smooth
    
    # Robust Sigma
    mad = np.nanmedian(np.abs(residuals - np.nanmedian(residuals)))
    sigma = 1.4826 * mad
    
    if not np.isfinite(sigma) or sigma == 0:
        sigma = np.nanstd(residuals) + 1e-12
        
    threshold = max(sigma_factor * sigma, min_threshold)
    mask = np.abs(residuals) >= threshold
    
    return mask, residuals, threshold

def combine_spike_masks(loop_mask, res_mask, intersect_spikes, spike_proximity):
    """
    Confirm spikes by intersecting loop and Hampel detections inside a proximity window.
    Returns the combined mask (intersection or union).
    """
    loop_mask = np.asarray(loop_mask, dtype=bool)
    res_mask = np.asarray(res_mask, dtype=bool)
    if not intersect_spikes:
        return loop_mask | res_mask

    prox = max(0, int(spike_proximity))
    final_mask = np.zeros_like(loop_mask, dtype=bool)
    loop_idx = np.where(loop_mask)[0]
    res_idx = np.where(res_mask)[0]

    if loop_idx.size == 0 or res_idx.size == 0:
        return final_mask

    for idx in loop_idx:
        if np.any(np.abs(res_idx - idx) <= prox):
            final_mask[idx] = True

    for idx in res_idx:
        if np.any(np.abs(loop_idx - idx) <= prox):
            final_mask[idx] = True

    return final_mask
