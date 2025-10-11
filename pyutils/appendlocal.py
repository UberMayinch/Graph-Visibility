import pandas as pd
import numpy as np
import struct

def _read_tsb1(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'TSB1':
            raise ValueError("Invalid TSB1 magic")
        cols = struct.unpack('<I', f.read(4))[0]
        rows = struct.unpack('<I', f.read(4))[0]
        if cols < 3:
            raise ValueError("TSB1 expects at least 3 columns")
        data = np.fromfile(f, dtype='<f8', count=rows * 3)
        data = data.reshape((-1, 3))
        df = pd.DataFrame(data, columns=['time', 'u', 'v'])
        return df

def AppendLocalOptima(csv_path, stabilizing_time, window_size, col_string):
    """Optimized local optima detection with vectorized operations."""
    # Read binary if .bin else CSV
    if str(csv_path).endswith('.bin'):
        df = _read_tsb1(csv_path)
    else:
        df = pd.read_csv(csv_path, engine='c')
    df = df.iloc[stabilizing_time:].copy()  # Use copy to avoid SettingWithCopyWarning

    # Vectorized rolling operations
    window = 2 * window_size + 1
    rolling_max = df[col_string].rolling(window, center=True, min_periods=1).max()
    rolling_min = df[col_string].rolling(window, center=True, min_periods=1).min()

    # Vectorized comparison
    df['is_local_opt'] = (df[col_string] == rolling_max) | (df[col_string] == rolling_min)
    
    return df
