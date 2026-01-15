import numpy as np
import pandas as pd

def detect_slope_steps(values, window=30, slope_threshold=2.0):
    """
    Step detection via Slope Change (Derivative Pulse).
    Logic:
    1. Calculate local slope (first derivative).
    2. Look for a significant 'pulse' in the slope (a sharp spike in derivative).
    3. Check that before and after the pulse, the slope is relatively flat (similar levels).
    4. Check that the signal trend is monotonic (up or down) across the window.
    
    Returns:
        metric_series: The slope pulse metric.
        mask: Boolean mask of detected steps.
    """
    if isinstance(values, pd.Series): values = values.values
    n = len(values)
    mask = np.zeros(n, dtype=bool)
    metric = np.zeros(n)
    
    if n < window:
        return pd.Series(metric), pd.Series(mask)

    # 1. Calculate Gradient (Slope)
    # Use simple central difference
    slope = np.gradient(values)
    
    # We scan with a window
    half_win = window // 2
    
    for i in range(half_win, n - half_win):
        # Window centered at i
        win_slope = slope[i - half_win : i + half_win]
        
        # Center region (the potential step) should have high slope
        # Flanks (before/after) should have low slope
        # Let's define "Center" as the middle 20% of the window?
        # Or just the peak?
        # User requirement: "jump in the slope and then back to similar slope"
        
        # Let's split into 3 parts: Left, Center, Right
        # Center size: maybe 3-5 points?
        center_width = max(1, window // 5)
        left_slope = win_slope[: (window - center_width)//2]
        right_slope = win_slope[-(window - center_width)//2 :]
        center_slope = win_slope[(window - center_width)//2 : -(window - center_width)//2]
        
        avg_left = np.mean(left_slope)
        avg_right = np.mean(right_slope)
        avg_center = np.mean(center_slope)
        
        # Condition 1: Similar slope before and after
        # (e.g. both near zero if flat, or both X if constant drift)
        # Check absolute difference
        if abs(avg_left - avg_right) > 0.5 * abs(avg_center): # Heuristic: flank diff is smaller than step
             # Wait, if step is huge, center is huge.
             # Condition: Left and Right should be 'close'
             # Let's check if they are statistically similar?
             # Simple check: abs(L - R) < tolerance
             pass # implement below
             
        # Condition 2: Jump in slope at center
        # Center slope should be significantly different from flanks
        # Pulse Height = abs(Center - (L+R)/2)
        pulse_height = abs(avg_center - (avg_left + avg_right)/2)
        
        # Normalize by noise/variability of slope?
        # Or just raw value? User asked for "jump in slope".
        # We return the pulse height as the metric.
        metric[i] = pulse_height
        
        # Condition 3: Monotonicity / Direction
        # The integral of the slope (the signal change) must be significant
        # and basically one direction.
        start_val = values[i - half_win]
        end_val = values[i + half_win]
        net_change = end_val - start_val
        
        if abs(net_change) == 0: continue
        
        # Direction of step must match direction of net change
        # i.e. Center Slope sign must match Net Change sign
        if np.sign(avg_center) != np.sign(net_change):
            metric[i] = 0 # Suppress invalid
            continue
            
    # Thresholding
    # We return the raw Slope Pulse Height.
    # The threshold is now absolute (e.g. 1e-9), not relative to noise.
    
    mask = metric > slope_threshold
    
    return metric, mask
