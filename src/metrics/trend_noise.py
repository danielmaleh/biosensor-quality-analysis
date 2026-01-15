import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline

def calculate_trend(series, window=50, method='median', poly_order=3, iterations=1, spline_s=None):
    """
    Calculates the signal trend using Median, Savitzky-Golay, or Spline.
    
    Parameters:
    - method: 'median', 'savgol', 'spline'
    - window: Used for Median/SG.
    - poly_order: Used for SG.
    - iterations: Repeats the filter (Median/SG only).
    - spline_s: Smoothing factor for UnivariateSpline. If None, defaults to len(series).
    """
    result = series.copy()
    
    # Special handling for Spline (Global fit, not iterative window)
    if method == 'spline':
        x = np.arange(len(series))
        # Handle NaNs for fitting
        mask = ~np.isnan(series)
        if mask.sum() < 4: return series # Too few points
        
        # Default s logic if not provided
        # s is the target sum of squared residuals.
        # User input 'spline_s' is likely a simplified factor (e.g. 0.1 to 1e6).
        s_val = spline_s if spline_s is not None else len(series)
        
        try:
            spl = UnivariateSpline(x[mask], series[mask], s=s_val)
            return pd.Series(spl(x), index=series.index)
        except:
            return series.rolling(window=window, center=True, min_periods=1).median()

    # Helper for single pass (Median/SG)
    def smooth_pass(s):
        if method == 'savgol':
            # Ensure window is odd and fits
            win = int(window)
            if win % 2 == 0: win += 1
            if win < poly_order + 2: win = poly_order + 2
            
            try:
                trend_array = savgol_filter(s.values, window_length=win, polyorder=poly_order)
                return pd.Series(trend_array, index=s.index)
            except:
                 # Fallback
                 return s.rolling(window=window, center=True, min_periods=1).median()
        else:
            # Median Filter (Default)
            return s.rolling(window=window, center=True, min_periods=1).median()

    # Iterative Smoothing
    for _ in range(max(1, int(iterations))):
        result = smooth_pass(result)
        # Handle NaNs that might creep in at edges
        result = result.fillna(method='bfill').fillna(method='ffill')
        
    return result

def calculate_noise_from_trend(series, trend, window=50):
    """
    Calculates Noise as the Rolling Median of Absolute Residuals (Raw - Trend).
    """
    residuals = series - trend
    abs_res = residuals.abs()
    # Rolling median of the absolute deviation
    return abs_res.rolling(window=window, center=True, min_periods=1).median()
