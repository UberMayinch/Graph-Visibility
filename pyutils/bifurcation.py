import matplotlib.pyplot as plt
import pandas as pd

def bifurcationdiagram(timeseries_dataframes, col="u", model="fhn", initial_conds="./"):
    all_peak_data = []
    stats_data = []
    
    for a, df in timeseries_dataframes.items():
        # Extract local optima for this 'a'
        peak_values = df[col][df["is_local_opt"]]
        for val in peak_values:
            all_peak_data.append({'A': a, 'u_peak': val})
        
        # Calculate statistics for this 'a'
        if len(peak_values) > 0:
            mean_val = peak_values.mean()
            std_val = peak_values.std()
            stats_data.append({
                'A': a, 
                'mean': mean_val, 
                'std': std_val,
                'upper_4std': mean_val + 4 * std_val,
            })

    # Convert to DataFrames
    peak_df = pd.DataFrame(all_peak_data)
    stats_df = pd.DataFrame(stats_data)

    # Plot
    plt.figure(figsize=(8, 6))
    
    # Plot scatter points
    for a in sorted(peak_df['A'].unique()):
        peaks = peak_df[peak_df['A'] == a]['u_peak']
        x_vals = [a] * len(peaks)
        plt.scatter(x_vals, peaks, alpha=0.1, s=1, color='black')

    # Plot mean and ±4σ lines
    if len(stats_df) > 0:
        plt.plot(stats_df['A'], stats_df['mean'], 'r-', linewidth=2, label='Mean')
        plt.plot(stats_df['A'], stats_df['upper_4std'], 'g--', linewidth=1, label='Mean + 4σ')
        # plt.plot(stats_df['A'], stats_df['lower_4std'], 'g--', linewidth=1, label='Mean - 4σ')

    plt.xlabel("Parameter value")
    plt.ylabel("Local optima of a")
    plt.title("Bifurcation Diagram for Timeseries")
    plt.grid(True)
    plt.legend()
    plt.savefig(f'plots/{model}/{initial_conds}/bifurcation.png')
    plt.close()

    return all_peak_data 


def bifurcation_smoothed(peak_data, model="fhn"):
    """
    Create a smoothed bifurcation diagram showing:
    1. Scatter plot from one time series realization
    2. Mean and mean+4σ lines for that specific time series
    3. Shaded regions showing statistics across all realizations
    """
    
    # Select the first realization for detailed plotting
    first_key = list(peak_data.keys())[0]
    selected_realization = peak_data[first_key]
    
    # Convert selected realization data to DataFrame for easier handling
    selected_df = pd.DataFrame(selected_realization)
    
    # Collect statistics from all realizations
    all_stats = []
    all_a_values = set()
    
    for (u0, v0), realization_data in peak_data.items():
        realization_df = pd.DataFrame(realization_data)
        
        # Group by parameter value A and calculate statistics
        for a in realization_df['A'].unique():
            peaks = realization_df[realization_df['A'] == a]['u_peak']
            if len(peaks) > 0:
                mean_val = peaks.mean()
                std_val = peaks.std()
                all_stats.append({
                    'A': a,
                    'u0': u0,
                    'v0': v0, 
                    'mean': mean_val,
                    'std': std_val,
                    'upper_4std': mean_val + 4 * std_val
                })
                all_a_values.add(a)
    
    # Convert to DataFrame
    all_stats_df = pd.DataFrame(all_stats)
    
    # Calculate overall statistics across all realizations for each A
    overall_stats = []
    for a in sorted(all_a_values):
        a_data = all_stats_df[all_stats_df['A'] == a]
        if len(a_data) > 0:
            overall_mean = a_data['mean'].mean()
            overall_upper = a_data['upper_4std'].mean()
            
            # Calculate min and max for mean and mean+4sd across realizations
            mean_min = a_data['mean'].min()
            mean_max = a_data['mean'].max()
            upper_4std_min = a_data['upper_4std'].min()
            upper_4std_max = a_data['upper_4std'].max()
            
            overall_stats.append({
                'A': a,
                'overall_mean': overall_mean,
                'overall_upper': overall_upper,
                'mean_min': mean_min,
                'mean_max': mean_max,
                'upper_4std_min': upper_4std_min,
                'upper_4std_max': upper_4std_max
            })
    
    overall_stats_df = pd.DataFrame(overall_stats)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # 1. Plot scatter points from selected realization (black, low opacity, small size)
    for a in sorted(selected_df['A'].unique()):
        peaks = selected_df[selected_df['A'] == a]['u_peak']
        x_vals = [a] * len(peaks)
        plt.scatter(x_vals, peaks, color='black', alpha=0.1, s=1, 
                   label='Selected realization' if a == sorted(selected_df['A'].unique())[0] else "")
    
    # 2. Calculate and plot mean and mean+4σ for selected realization
    selected_stats = []
    for a in sorted(selected_df['A'].unique()):
        peaks = selected_df[selected_df['A'] == a]['u_peak']
        if len(peaks) > 0:
            mean_val = peaks.mean()
            std_val = peaks.std()
            selected_stats.append({
                'A': a,
                'mean': mean_val,
                'upper_4std': mean_val + 4 * std_val
            })
    
    selected_stats_df = pd.DataFrame(selected_stats)
    
    # 3. Plot shaded regions for min-max ranges across all realizations
    if len(overall_stats_df) > 0:
        # Shaded region for mean (min to max across realizations)
        plt.fill_between(overall_stats_df['A'], 
                        overall_stats_df['mean_min'],
                        overall_stats_df['mean_max'],
                        color='red', alpha=0.2, label='Mean range (all realizations)')
        
        # Shaded region for mean + 4σ (min to max across realizations)
        plt.fill_between(overall_stats_df['A'],
                        overall_stats_df['upper_4std_min'],
                        overall_stats_df['upper_4std_max'],
                        color='green', alpha=0.2, label='Mean + 4σ range (all realizations)')
        
        # Plot mean lines for all realizations
        plt.plot(overall_stats_df['A'], overall_stats_df['overall_mean'],
                'r--', linewidth=1, alpha=0.7, label='Overall mean (all realizations)')
        plt.plot(overall_stats_df['A'], overall_stats_df['overall_upper'],
                'g--', linewidth=1, alpha=0.7, label='Overall mean + 4σ (all realizations)')
    
    plt.xlabel("Parameter value (A)")
    plt.ylabel("Local optima")
    plt.title("Smoothed Bifurcation Diagram")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'plots/{model}/bifurcation_smoothed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return overall_stats_df