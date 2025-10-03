import random
import matplotlib.pyplot as plt
import pandas as pd

def PlotExtrema(df, a, col_string1, col_string2, model, initial_conds="./"): 
    plt.figure(figsize=(12, 6))

    mean_u = df[col_string1].mean()
    std_u = df[col_string1].std()

    plt.subplot(2, 1, 1)
    plt.plot(df['time'], df[col_string1], label=col_string1+'(t)')
    plt.title(f'{col_string1} trajectory over time, A = {a}')
    plt.xlabel('Time')
    plt.ylabel(col_string1)

    # Plot local optima
    plt.scatter(
        df.loc[df['is_local_opt'], 'time'],
        df.loc[df['is_local_opt'], col_string1],
        color='red', marker='o', label='Local Optima', zorder=3
    )

    # Plot mean and ±6 std lines
    plt.axhline(mean_u, color='green', linestyle='--', label='Mean')
    plt.axhline(mean_u + 6 * std_u, color='orange', linestyle='--', label='Mean ± 6σ')
    plt.axhline(mean_u - 6 * std_u, color='orange', linestyle='--')

    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(df['time'], df[col_string2], label = col_string2 + '(t)')
    plt.title(f'{col_string2} trajectory over time, A = {a}')
    plt.xlabel('Time')
    plt.ylabel(col_string2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/{model}/{initial_conds}/extrema_{a}.png')
    plt.close()
    return

def process_timeseries(A, timeseries_dataframes, col_string1, col_string2, model, initial_conds="./", mode='all', n=None, subset=None):

    """
    Parameters:
    - A: list of identifiers (same length as timeseries_dataframes)
    - timeseries_dataframes: list of DataFrames corresponding to A
    - mode: 'all', 'random', or 'subset'
    - n: number of samples to select (used if mode == 'random')
    - subset: list of values to use from A (used if mode == 'subset')
    """

    # Convert to list of tuples for easier handling
    data_pairs = list(zip(A, timeseries_dataframes))

    if mode == 'all':
        selected = data_pairs

    elif mode == 'random':
        if n is None:
            raise ValueError("You must specify 'n' when using mode='random'")
        if n > len(A):
            raise ValueError(f"n={n} is greater than number of available elements={len(A)}")
        selected = random.sample(data_pairs, n)

    elif mode == 'subset':
        if subset is None:
            raise ValueError("You must provide a subset list when using mode='subset'")
        # Keep only those entries where a ∈ subset
        selected = [pair for pair in data_pairs if pair[0] in subset]

    else:
        raise ValueError("mode must be one of: 'all', 'random', or 'subset'")

    for a, df in selected:
        PlotExtrema(df, a, col_string1, col_string2, model, initial_conds=initial_conds)

