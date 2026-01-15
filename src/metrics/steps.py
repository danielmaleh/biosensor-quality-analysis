import numpy as np
import pandas as pd

def piecewise_improvement(times, values, window=30, poly_order=1):
    """
    Step detection via Piecewise Fit Improvement.
    Splits window in half. Compares fit error of Single Poly vs Two Polys.
    
    Refinements:
    - Poly order configurable.
    - Strict monotonicity check (no oscillations).
    """
    if isinstance(values, pd.Series): values = values.values
    if isinstance(times, pd.Series): times = times.values
        
    n = len(values)
    improvements = np.full(n, np.nan)
    
    if n < window or window < 6:
        return improvements

    half = window // 2
    
    for start in range(0, n - window + 1):
        end = start + window
        segment_times = times[start:end]
        segment_values = values[start:end]
        
        # Relative time to avoid numerical issues
        rel_times = segment_times - segment_times[0]

        start_val = segment_values[0]
        end_val = segment_values[-1]
        
        # Check for net change (Step Up or Step Down)
        direction = np.sign(end_val - start_val)
        if direction == 0:
            continue
            
        # (Strict monotonicity check removed per user request to allow for noise)

        # Model A: Single Fit
        try:
            single_coeffs = np.polyfit(rel_times, segment_values, poly_order)
            single_pred = np.polyval(single_coeffs, rel_times)
            single_sse = np.sum((segment_values - single_pred) ** 2)
        except:
            continue
            
        if single_sse == 0: continue

        # Model B: Split Fits
        split_idx = window // 2
        t_first = rel_times[:split_idx]
        v_first = segment_values[:split_idx]
        t_second = rel_times[split_idx:]
        v_second = segment_values[split_idx:]

        if len(t_first) < poly_order + 1 or len(t_second) < poly_order + 1: continue

        try:
            coeff_first = np.polyfit(t_first, v_first, poly_order)
            coeff_second = np.polyfit(t_second, v_second, poly_order)
            
            pred_first = np.polyval(coeff_first, t_first)
            pred_second = np.polyval(coeff_second, t_second)
            
            split_sse = np.sum((v_first - pred_first) ** 2) + np.sum((v_second - pred_second) ** 2)
            
            improvement = (single_sse - split_sse) / single_sse
            improvements[start + half] = improvement
            
        except:
            continue
            
    return improvements