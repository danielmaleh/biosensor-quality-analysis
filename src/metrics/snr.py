import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def calculate_snr_stats(series, noise_series, spline_window=50, method='median', poly_order=3):
    """
    Calculates SNR based on: Smooth Signal / Noise(MAD).
    
    Parameters:
    - method: 'median' (Robust to steps) or 'savgol' (Smoother, polynomial).
    
    Returns:
        smooth (pd.Series): The signal trend.
        snr (pd.Series): The SNR over time.
    """
    # 1. Calculate Smooth Trend
    if method == 'savgol':
        # Ensure window is odd and fits
        win = spline_window
        if win % 2 == 0: win += 1
        if win < poly_order + 2: win = poly_order + 2
        
        try:
            trend_array = savgol_filter(series.values, window_length=win, polyorder=poly_order)
            smooth = pd.Series(trend_array, index=series.index)
        except:
             # Fallback if window too small for data
             smooth = series.rolling(window=spline_window, center=True, min_periods=1).median()
    else:
        # Median Filter (Default)
        smooth = series.rolling(window=spline_window, center=True, min_periods=1).median()
    
    # 2. SNR = Signal / Noise
    # Add epsilon to avoid div by zero
    snr = smooth / (noise_series + 1e-12)
    
    return smooth, snr