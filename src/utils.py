import pandas as pd
import re

def load_data(file_path):
    """
    Loads the biochemical CSV data.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Identify Time Column
        # Heuristic: Contains 'time' or 'timestamp', or is the first non-index column.
        time_col = None
        for col in df.columns:
            if 'time' in col.lower():
                time_col = col
                break
        
        if not time_col:
            # Fallback: Use the first column
            time_col = df.columns[1] # 0 might be index
            
        # Identify Sensor Columns
        # Filter for Signal Mologram (SM_R#C#) columns, excluding timestamp columns
        sensor_cols = [c for c in df.columns if re.match(r"SM_R\d+C\d+", c) and "timestamp" not in c.lower()]
        
        return df, time_col, sensor_cols
        
    except Exception as e:
        return None, None, None

import numpy as np

def normalize_minmax(series):
    """Normalizes series to [0, 1] range."""
    s_min = np.nanmin(series)
    s_max = np.nanmax(series)
    if s_max - s_min == 0:
        return np.zeros_like(series)
    return (series - s_min) / (s_max - s_min)

def normalize_zscore(series):
    """Standardizes series (mean=0, std=1)."""
    mean = np.nanmean(series)
    std = np.nanstd(series)
    if std == 0:
        return np.zeros_like(series)
    return (series - mean) / std
