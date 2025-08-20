# %%
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import numpy as np
import random
from collections import defaultdict
import os

# %%
compile_cmd = "make all"
subprocess.run(compile_cmd, shell=True, check=True)
dir_cmd = "make dirs"
subprocess.run(dir_cmd, shell=True, check=True)

# %%
np.random.seed(42)
stabilizing_time = 2000
window_size = 1
precision_degree = 6
num_params = 21

# %% [markdown]
# ## Helper Functions
# ---

# %%
def AppendLocalOptima(csv_path, stabilizing_time, window_size, col_string):
    df = pd.read_csv(csv_path)
    df = df.iloc[stabilizing_time:]

    rolling_max = df[col_string].rolling(2 * window_size+ 1, center=True).max()
    rolling_min = df[col_string].rolling(2 * window_size+ 1, center=True).min()

    df['is_local_opt'] = (df[col_string] == rolling_max) | (df[col_string] == rolling_min)
    return df



# %%

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


# %%

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



# %%

def bifurcationdiagram(timeseries_dataframes, col="u"):
    all_peak_data = []
    for a, df in timeseries_dataframes.items():
        # Extract local optima for this 'a'
        peak_values = df[col][df["is_local_opt"]]
        for val in peak_values:
            all_peak_data.append({'A': a, 'u_peak': val})

    # Convert to DataFrame
    peak_df = pd.DataFrame(all_peak_data)

    # Plot
    plt.figure(figsize=(8, 6))
    for a in sorted(peak_df['A'].unique()):
        peaks = peak_df[peak_df['A'] == a]['u_peak']
        x_vals = [a] * len(peaks)
        plt.scatter(x_vals, peaks, label=f"A = {a}", alpha=0.2)

    plt.xlabel("Parameter value")
    plt.ylabel("Local optima of a")
    plt.title("Bifurcation Diagram for Timeseries")
    plt.grid(True)
    plt.show()

    return 

# %%

def plotGraphMetrics(x, y, model='linard'):
    for x0, y0 in zip(x, y):
        subprocess.run(f"./swap_uv.sh data/{model}/{x0}_{y0}", shell=True, check=True)
        command = f"./graph_characteristics data/{model}/{x0}_{y0}/"
        subprocess.run(command, shell=True, check=True)
        
        # Read the generated graph metrics file
        graph_metrics = pd.read_csv(f'data/{model}/{x0}_{y0}/graph_metrics.csv')

        # Sort by parameter column
        graph_metrics = graph_metrics.sort_values('parameter')

        # Convert parameter column to numeric for proper plotting
        graph_metrics['parameter'] = pd.to_numeric(graph_metrics['parameter'])

        # Create line plots
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(graph_metrics['parameter'], graph_metrics['max_degree'], marker='o', markersize=3)
        plt.xlabel('Parameter')
        plt.ylabel('Max Degree')
        plt.title(f'Parameter vs Max Degree (x0={x0:.6f}, y0={y0:.6f})')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(graph_metrics['parameter'], graph_metrics['avg_degree'], marker='o', markersize=3)
        plt.xlabel('Parameter')
        plt.ylabel('Average Degree')
        plt.title(f'Parameter vs Average Degree (x0={x0:.6f}, y0={y0:.6f})')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'plots/{model}/{x0}_{y0}/graph_metrics.png')
        plt.show()

        subprocess.run(f"./swap_uv.sh data/{model}/{x0}_{y0}", shell=True, check=True)

# %% [markdown]
# # FHN Model
# ---

# %%
u = np.random.random(size=10)*0.001 + -0.2
v = np.random.random(size = 10)*0.0001 + -0.002
u = np.round(u, precision_degree)
v = np.round(v, precision_degree)
print(f"u values: {u}")
print(f"v values: {v}")

# %%
A = np.round(np.linspace(0.62, 0.63, num_params), precision_degree) 
 

print(A)

csv_paths_fhn = defaultdict(list)
    
for u0, v0 in zip(u, v):
    csv_paths_fhn[(u0, v0)] = {
        a: f"data/fhn/{u0}_{v0}/output_{a}.csv"
        for a in A
    }

print(csv_paths_fhn.items())


# %%
for u0, v0 in zip(u, v):
    make_initial_cond_dir = f"mkdir -p -- data/fhn/{u0}_{v0}"
    print(make_initial_cond_dir)
    subprocess.run(make_initial_cond_dir, shell=True)
    for a in A:
        subprocess.run(f"./fhn {a} {u0} {v0}", shell=True, check=True)
    

# %%
fhn_dataframes = defaultdict(dict)

for (u0, v0), a_paths in csv_paths_fhn.items():
    for a, path in a_paths.items():
        fhn_dataframes[(u0, v0)][a] = AppendLocalOptima(
            path, stabilizing_time, window_size, 'u'
        )

# print(fhn_dataframes.values())


# %%

for (u0, v0) in zip(u, v):
    # Sort dataframes by A values (keys)
    sorted_items = sorted(fhn_dataframes[(u0, v0)].items())
    # print([(a, df) for a, df in sorted_items]) 
    dataframes_list = [df for a, df in sorted_items]
    print(f"Number of dataframes for u0={u0:.6f}, v0={v0:.6f}: {len(dataframes_list)}")
    
    make_initial_cond_dir = f"mkdir -p -- plots/fhn/{u0}_{v0}"
    print(make_initial_cond_dir)
    subprocess.run(make_initial_cond_dir, shell=True)
    process_timeseries(A, dataframes_list, 'u', 'v', 'fhn', mode='all', initial_conds=f"./{u0}_{v0}")


# %%
for (u0, v0) in zip(u, v):
    print(f"Initial conditions: u0 = {u0}, v0 = {v0}")
    bifurcationdiagram(fhn_dataframes[(u0, v0)])

# %%
plotGraphMetrics(u, v, "fhn")

# %% [markdown]
# # Linard Model
# ---

# %%
x = np.random.random(size=10) - 0.5
y = np.random.random(size = 10)
x = np.round(x, precision_degree)
y = np.round(y, precision_degree)

# %%

omega_vals = np.round(np.linspace(0.64, 0.74, num_params), precision_degree)

for x0, y0 in zip(x, y):
    make_initial_cond_dir = f"mkdir -p -- data/linard/{x0}_{y0}"
    print(make_initial_cond_dir)
    subprocess.run(make_initial_cond_dir, shell=True)
    for omega in omega_vals:
        subprocess.run(f"./linard {omega} {x0} {y0}", shell=True, check=True)

csv_paths_linard = defaultdict(list)
    
for x0, y0 in zip(x, y):
    csv_paths_linard[(x0, y0)] = {
        omega: f"data/linard/{x0}_{y0}/output_{omega}.csv"
        for omega in omega_vals
    }

print(csv_paths_fhn.items())

# %%
linard_dataframes = defaultdict(dict)

for (x0, y0), omega_paths in csv_paths_linard.items():
    for omega, path in omega_paths.items():
        linard_dataframes[(x0, y0)][omega] = AppendLocalOptima(
            path, stabilizing_time, window_size, 'x'
        )

# %%

for (x0, y0) in zip(x, y):
    # Sort dataframes by A values (keys)
    sorted_items = sorted(linard_dataframes[(x0, y0)].items())
    dataframes_list = [df for omega, df in sorted_items]
    print(f"Number of dataframes for x0={x0:.6f}, y0={y0:.6f}: {len(dataframes_list)}")
    
    make_initial_cond_dir = f"mkdir -p -- plots/linard/{x0}_{y0}"
    subprocess.run(make_initial_cond_dir, shell=True)
    process_timeseries(A, dataframes_list, 'x', 'y', 'linard', mode='all', initial_conds=f"./{x0}_{y0}")

# %%
for (x0, y0) in zip(x, y):
    print(f"Initial conditions: x0 = {x0}, y0 = {y0}")
    bifurcationdiagram(linard_dataframes[(x0, y0)], col="x")


# %%

plotGraphMetrics(x, y, "linard")


