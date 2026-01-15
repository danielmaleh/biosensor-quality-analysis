import pandas as pd
import numpy as np

def calculate_rolling_mad(series, window=50):
    """
    Calculates the Rolling Mean Absolute Deviation (MAD).
    MAD = median( | x - median(x) | )
    """
    # Helper for rolling apply (can be slow, but consistent with example)
    # Alternatively, use the optimized approximation from previous step if speed is key.
    # The example uses .apply(mad_func, raw=True)
    
    def mad_func(x):
        median = np.median(x)
        return np.median(np.abs(x - median))

    return series.rolling(window=window, center=True, min_periods=1).apply(mad_func, raw=True)