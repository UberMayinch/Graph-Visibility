# %%
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import numpy as np
import random
from collections import defaultdict
import os

# %%
# Compile and setup directories
compile_cmd = "make all"
subprocess.run(compile_cmd, shell=True, check=True)
dir_cmd = "make dirs"
subprocess.run(dir_cmd, shell=True, check=True)

# %%
np.random.seed(42)
stabilizing_time = 2000
window_size = 1

# %% [markdown]
# ## Helper Functions
# ---

# %%
def AppendLocalOptima(csv_path, stabilizing_time, window_size, col_string):
    """Load CSV and identify local optima in specified column"""
    if not os.path.exists(csv_path):
        print(f"Warning: File {csv_path} does not exist")
        return None
        
    df = pd.read_csv(csv_path)
    if len(df) <= stabilizing_time:
        print(f"Warning: File {csv_path} has insufficient data")
        return None
        
    df = df.iloc[stabilizing_time:]
    df = df.reset_index(drop=True)  # Reset index after slicing

    # Calculate rolling max and min
    rolling_max = df[col_string].rolling(2 * window_size + 1, center=True).max()
    rolling_min = df[col_string].rolling(2 * window_size + 1, center=True).min()

    df['is_local_opt'] = (df[col_string] == rolling_max) | (df[col_string] == rolling_min)
    return df

# %%
def PlotExtrema(df, param_val, col_string1, col_string2, model, initial_conds="./"):
    """Plot time series with local optima marked"""
    if df is None or df.empty:
        print(f"Warning: No data to plot for parameter {param_val}")
        return
        
    plt.figure(figsize=(12, 6))

    mean_u = df[col_string1].mean()
    std_u = df[col_string1].std()

    plt.subplot(2, 1, 1)
    plt.plot(df['time'], df[col_string1], label=f'{col_string1}(t)', linewidth=0.8)
    plt.title(f'{col_string1} trajectory over time, Parameter = {param_val:.4f}')
    plt.xlabel('Time')
    plt.ylabel(col_string1)

    # Plot local optima
    local_opt_mask = df['is_local_opt'] & df[col_string1].notna()
    if local_opt_mask.any():
        plt.scatter(
            df.loc[local_opt_mask, 'time'],
            df.loc[local_opt_mask, col_string1],
            color='red', marker='o', s=10, label='Local Optima', zorder=3
        )

    # Plot mean and ±6 std lines
    plt.axhline(mean_u, color='green', linestyle='--', label='Mean', alpha=0.7)
    plt.axhline(mean_u + 6 * std_u, color='orange', linestyle='--', label='Mean ± 6σ', alpha=0.7)
    plt.axhline(mean_u - 6 * std_u, color='orange', linestyle='--', alpha=0.7)

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(df['time'], df[col_string2], label=f'{col_string2}(t)', linewidth=0.8, color='orange')
    plt.title(f'{col_string2} trajectory over time, Parameter = {param_val:.4f}')
    plt.xlabel('Time')
    plt.ylabel(col_string2)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    plot_dir = f'plots/{model}/{initial_conds}'
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.savefig(f'{plot_dir}/extrema_{param_val:.6f}.png', dpi=150, bbox_inches='tight')
    plt.close()

# %%
def process_timeseries(param_values, timeseries_dataframes, col_string1, col_string2, model, initial_conds="./", mode='all', n=None, subset=None):
    """Process and plot time series data"""
    # Filter out None dataframes
    valid_data = [(p, df) for p, df in zip(param_values, timeseries_dataframes) if df is not None]
    
    if not valid_data:
        print("No valid data to process")
        return
    
    if mode == 'all':
        selected = valid_data
    elif mode == 'random':
        if n is None or n > len(valid_data):
            n = min(10, len(valid_data))  # Default to 10 or all available
        selected = random.sample(valid_data, n)
    elif mode == 'subset':
        if subset is None:
            subset = [valid_data[0][0]]  # Default to first parameter
        selected = [pair for pair in valid_data if any(abs(pair[0] - s) < 1e-6 for s in subset)]
    else:
        raise ValueError("mode must be one of: 'all', 'random', or 'subset'")

    print(f"Processing {len(selected)} time series plots for {model}")
    for param_val, df in selected:
        PlotExtrema(df, param_val, col_string1, col_string2, model, initial_conds=initial_conds)

# %%
def bifurcation_diagram(param_values, timeseries_dataframes, col_string, model, initial_conds="./"):
    """Create bifurcation diagram from local optima"""
    all_peak_data = []

    for param_val, df in zip(param_values, timeseries_dataframes):
        if df is None:
            continue
            
        # Extract local optima for this parameter value
        local_opt_mask = df["is_local_opt"] & df[col_string].notna()
        peak_values = df[local_opt_mask][col_string]
        
        for val in peak_values:
            all_peak_data.append({'param': param_val, f'{col_string}_peak': val})

    if not all_peak_data:
        print("No peak data available for bifurcation diagram")
        return

    # Convert to DataFrame
    peak_df = pd.DataFrame(all_peak_data)

    # Plot
    plt.figure(figsize=(12, 8))
    
    # Use scatter plot with small alpha for better visibility
    plt.scatter(peak_df['param'], peak_df[f'{col_string}_peak'], 
                s=1, alpha=0.6, c='blue')
    
    plt.xlabel("Parameter Value")
    plt.ylabel(f"Local Optima of {col_string}")
    plt.title(f"Bifurcation Diagram: {model.upper()} Model")
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_dir = f'plots/{model}/{initial_conds}'
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f'{plot_dir}/bifurcation_diagram.png', dpi=150, bbox_inches='tight')
    plt.show()

# %%
def run_visibility_graph_analysis(model, data_dir):
    """Run visibility graph analysis using C++ executable"""
    try:
        command = f"./graph_metrics {data_dir}"
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"Visibility graph analysis completed for {model}")
        
        # Check if output file exists
        if os.path.exists('graph_metrics.csv'):
            return plot_graph_metrics(model)
        else:
            print("Warning: graph_metrics.csv not found")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error running visibility graph analysis: {e}")
        return None

# %%
def plot_graph_metrics(model):
    """Plot graph metrics from visibility graph analysis"""
    if not os.path.exists('graph_metrics.csv'):
        print("graph_metrics.csv not found")
        return None
        
    graph_metrics = pd.read_csv('graph_metrics.csv')
    
    # Sort by parameter column
    graph_metrics = graph_metrics.sort_values('parameter')
    graph_metrics['parameter'] = pd.to_numeric(graph_metrics['parameter'])

    # Create line plots
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(graph_metrics['parameter'], graph_metrics['max_degree'], 
             marker='o', markersize=2, linewidth=1)
    plt.xlabel('Parameter')
    plt.ylabel('Max Degree')
    plt.title(f'{model.upper()}: Parameter vs Max Degree')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(graph_metrics['parameter'], graph_metrics['avg_degree'], 
             marker='o', markersize=2, linewidth=1, color='orange')
    plt.xlabel('Parameter')
    plt.ylabel('Average Degree')
    plt.title(f'{model.upper()}: Parameter vs Average Degree')
    plt.grid(True, alpha=0.3)
    
    # Add clustering coefficient if available
    if 'clustering_coeff' in graph_metrics.columns:
        plt.subplot(1, 3, 3)
        plt.plot(graph_metrics['parameter'], graph_metrics['clustering_coeff'], 
                 marker='o', markersize=2, linewidth=1, color='green')
        plt.xlabel('Parameter')
        plt.ylabel('Clustering Coefficient')
        plt.title(f'{model.upper()}: Parameter vs Clustering Coefficient')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save plot
    os.makedirs(f'plots/{model}', exist_ok=True)
    plt.savefig(f'plots/{model}/graph_metrics.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return graph_metrics

# %% [markdown]
# # FitzHugh-Nagumo (FHN) Model
# ---

# %%
print("=== FitzHugh-Nagumo Model Simulation ===")

# Generate initial conditions
np.random.seed(42)
u_init = np.random.random(size=5) * 0.001 - 0.2  # Reduced to 5 for faster testing
v_init = np.random.random(size=5) * 0.0001 - 0.002

print(f"Initial conditions:")
for i, (u0, v0) in enumerate(zip(u_init, v_init)):
    print(f"  IC {i+1}: u0={u0:.6f}, v0={v0:.6f}")

# %%
# Parameter range for FHN
A_fhn = np.linspace(0.62, 0.63, 21)  # Reduced for faster testing
print(f"Parameter range: A ∈ [{A_fhn[0]:.3f}, {A_fhn[-1]:.3f}] with {len(A_fhn)} values")

# Run FHN simulations
print("Running FHN simulations...")
fhn_dataframes = {}

for i, (u0, v0) in enumerate(zip(u_init, v_init)):
    ic_key = f"{u0:.6f}_{v0:.6f}"
    fhn_dataframes[ic_key] = {}
    
    print(f"  Running simulations for IC {i+1}/{len(u_init)}")
    
    for j, a in enumerate(A_fhn):
        # Run simulation
        cmd = f"./fhn {a} {u0} {v0}"
        try:
            subprocess.run(cmd, shell=True, check=True)
            
            # Load and process data
            csv_path = f"data/fhn/{u0:.6f}_{v0:.6f}_output_{a}.csv"
            df = AppendLocalOptima(csv_path, stabilizing_time, window_size, 'u')
            fhn_dataframes[ic_key][a] = df
            
        except subprocess.CalledProcessError as e:
            print(f"    Error running simulation for a={a}: {e}")
            fhn_dataframes[ic_key][a] = None
        
        if (j + 1) % 5 == 0:
            print(f"    Completed {j+1}/{len(A_fhn)} parameter values")

# %%
# Process FHN results
print("Processing FHN results...")

for ic_key in fhn_dataframes.keys():
    u0, v0 = ic_key.split('_')
    
    # Create plots directory for this initial condition
    plot_dir = f"plots/fhn/{ic_key}"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Extract dataframes in parameter order
    sorted_items = sorted(fhn_dataframes[ic_key].items())
    param_values = [a for a, df in sorted_items]
    dataframes_list = [df for a, df in sorted_items]
    
    valid_count = sum(1 for df in dataframes_list if df is not None)
    print(f"  IC {ic_key}: {valid_count}/{len(dataframes_list)} valid simulations")
    
    if valid_count > 0:
        # Generate sample plots
        process_timeseries(param_values, dataframes_list, 'u', 'v', 'fhn', 
                         initial_conds=ic_key, mode='random', n=3)
        
        # Create bifurcation diagram for this IC
        bifurcation_diagram(param_values, dataframes_list, 'u', 'fhn', 
                          initial_conds=ic_key)

# %%
# FHN Visibility Graph Analysis
print("Running FHN visibility graph analysis...")
fhn_metrics = run_visibility_graph_analysis('fhn', 'data/fhn')

# %% [markdown]
# # Liénard Model
# ---

# %%
print("\n=== Liénard Model Simulation ===")

# Generate initial conditions for Liénard
np.random.seed(42)
x_init = np.random.random(size=5) - 0.5  # Reduced to 5 for faster testing
y_init = np.random.random(size=5)

print(f"Initial conditions:")
for i, (x0, y0) in enumerate(zip(x_init, y_init)):
    print(f"  IC {i+1}: x0={x0:.6f}, y0={y0:.6f}")

# %%
# Parameter range for Liénard
omega_vals = np.linspace(0.64, 0.74, 21)  # Reduced for faster testing
print(f"Parameter range: ω ∈ [{omega_vals[0]:.3f}, {omega_vals[-1]:.3f}] with {len(omega_vals)} values")

# Run Liénard simulations
print("Running Liénard simulations...")
linard_dataframes = {}

for i, (x0, y0) in enumerate(zip(x_init, y_init)):
    ic_key = f"{x0:.6f}_{y0:.6f}"
    linard_dataframes[ic_key] = {}
    
    print(f"  Running simulations for IC {i+1}/{len(x_init)}")
    
    for j, omega in enumerate(omega_vals):
        # Run simulation
        cmd = f"./linard {omega} {x0} {y0}"
        try:
            subprocess.run(cmd, shell=True, check=True)
            
            # Load and process data
            csv_path = f"data/linard/output_{omega}_{x0:.6f}_{y0:.6f}.csv"
            df = AppendLocalOptima(csv_path, stabilizing_time, window_size, 'x')
            linard_dataframes[ic_key][omega] = df
            
        except subprocess.CalledProcessError as e:
            print(f"    Error running simulation for ω={omega}: {e}")
            linard_dataframes[ic_key][omega] = None
        
        if (j + 1) % 5 == 0:
            print(f"    Completed {j+1}/{len(omega_vals)} parameter values")

# %%
# Process Liénard results
print("Processing Liénard results...")

for ic_key in linard_dataframes.keys():
    x0, y0 = ic_key.split('_')
    
    # Create plots directory for this initial condition
    plot_dir = f"plots/linard/{ic_key}"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Extract dataframes in parameter order
    sorted_items = sorted(linard_dataframes[ic_key].items())
    param_values = [omega for omega, df in sorted_items]
    dataframes_list = [df for omega, df in sorted_items]
    
    valid_count = sum(1 for df in dataframes_list if df is not None)
    print(f"  IC {ic_key}: {valid_count}/{len(dataframes_list)} valid simulations")
    
    if valid_count > 0:
        # Generate sample plots
        process_timeseries(param_values, dataframes_list, 'x', 'y', 'linard', 
                         initial_conds=ic_key, mode='random', n=3)
        
        # Create bifurcation diagram for this IC
        bifurcation_diagram(param_values, dataframes_list, 'x', 'linard', 
                          initial_conds=ic_key)

# %%
# Liénard Visibility Graph Analysis
print("Running Liénard visibility graph analysis...")
linard_metrics = run_visibility_graph_analysis('linard', 'data/linard')

# %% [markdown]
# # Summary and Analysis
# ---

# %%
print("\n=== Simulation Summary ===")
print("All simulations completed!")
print("\nGenerated outputs:")
print("- Time series plots with local optima marked")
print("- Bifurcation diagrams showing parameter dependence")
print("- Visibility graph metrics analysis")
print("\nCheck the following directories:")
print("- plots/fhn/ : FHN model results")
print("- plots/linard/ : Liénard model results")
print("- data/fhn/ : FHN simulation data")
print("- data/linard/ : Liénard simulation data")

# Display final metrics if available
if 'fhn_metrics' in locals() and fhn_metrics is not None:
    print(f"\nFHN Graph Metrics Summary:")
    print(f"- Max degree range: [{fhn_metrics['max_degree'].min():.2f}, {fhn_metrics['max_degree'].max():.2f}]")
    print(f"- Avg degree range: [{fhn_metrics['avg_degree'].min():.2f}, {fhn_metrics['avg_degree'].max():.2f}]")

if 'linard_metrics' in locals() and linard_metrics is not None:
    print(f"\nLiénard Graph Metrics Summary:")
    print(f"- Max degree range: [{linard_metrics['max_degree'].min():.2f}, {linard_metrics['max_degree'].max():.2f}]")
    print(f"- Avg degree range: [{linard_metrics['avg_degree'].min():.2f}, {linard_metrics['avg_degree'].max():.2f}]")