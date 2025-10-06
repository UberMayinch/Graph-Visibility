import pandas as pd
import numpy as np

def AppendLocalOptima(csv_path, stabilizing_time, window_size, col_string):
    """Optimized local optima detection with vectorized operations."""
    # Use faster CSV reading
    df = pd.read_csv(csv_path, engine='c')
    df = df.iloc[stabilizing_time:].copy()  # Use copy to avoid SettingWithCopyWarning

    # Vectorized rolling operations
    window = 2 * window_size + 1
    rolling_max = df[col_string].rolling(window, center=True, min_periods=1).max()
    rolling_min = df[col_string].rolling(window, center=True, min_periods=1).min()

    # Vectorized comparison
    df['is_local_opt'] = (df[col_string] == rolling_max) | (df[col_string] == rolling_min)
    
    return df
