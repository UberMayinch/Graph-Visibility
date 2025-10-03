import pandas as pd

def AppendLocalOptima(csv_path, stabilizing_time, window_size, col_string):
    df = pd.read_csv(csv_path)
    df = df.iloc[stabilizing_time:]

    rolling_max = df[col_string].rolling(2 * window_size+ 1, center=True).max()
    rolling_min = df[col_string].rolling(2 * window_size+ 1, center=True).min()

    df['is_local_opt'] = (df[col_string] == rolling_max) | (df[col_string] == rolling_min)
    return df
