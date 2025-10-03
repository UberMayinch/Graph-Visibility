import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import csv
from glob import glob

def swap_columns(folder_path):
    """Swap u and v columns in CSV files (replacement for swap_uv.sh)"""
    csv_files = glob(os.path.join(folder_path, "output_*.csv"))
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'u' in df.columns and 'v' in df.columns:
                df['u'], df['v'] = df['v'], df['u']
                df.to_csv(csv_file, index=False)
        except Exception as e:
            print(f"Warning: Could not swap columns in {csv_file}: {e}")

def constructGraphs(x, y, model='linard'):
    for x0, y0 in zip(x, y):
        folder_path = f"data/{model}/{x0}_{y0}"
        
        # Construct graphs (no column swapping needed - C++ code reads correct columns)
        weighted_command = f"./weighted_construct {folder_path}/"
        unweighted_command = f"./unweighted_construct {folder_path}/"
        subprocess.run(weighted_command, shell=True, check=True)
        subprocess.run(unweighted_command, shell=True, check=True)

def summarizeWeightedDegrees(x, y, model='linard', output_name='weighted_degree_stats.csv'):
    for x0, y0 in zip(x, y):
        folder = f"data/{model}/{x0}_{y0}"
        # Exclude _metrics.csv files from the graph files list
        files = sorted([f for f in glob(os.path.join(folder, "weighted_graph_*.csv")) 
                       if not f.endswith('_metrics.csv')])
        rows = []
        print(folder)
        for fp in files:
            m = re.search(r'weighted_graph_(.+)\.csv$', os.path.basename(fp))
            param = m.group(1) if m else os.path.basename(fp)
            
            # Initialize metrics with default values
            metrics = {
                'param': param,
                'avg_degree': 0.0,
                'max_degree': 0.0,
                'degree_entropy': 0.0,
                'avg_path_length': 0.0,
                'clustering_coefficient': 0.0,
                'density': 0.0,
                'mean_degree': 0.0,
                'median_degree': 0.0,
                'min_degree': 0.0,
                'max_degree_stats': 0.0
            }
            
            # Read advanced metrics if available
            metrics_file = fp.replace('.csv', '_metrics.csv')
            if os.path.exists(metrics_file):
                try:
                    metrics_df = pd.read_csv(metrics_file)
                    for _, row in metrics_df.iterrows():
                        metric_name = row['metric']
                        if metric_name in metrics:
                            metrics[metric_name] = float(row['value'])
                        elif metric_name == 'max_degree':
                            metrics['max_degree_stats'] = float(row['value'])
                except Exception as e:
                    print(f"Warning: Could not read metrics from {metrics_file}: {e}")
            
            # Calculate basic degree statistics as fallback
            df = pd.read_csv(fp, header=None, comment='#', skip_blank_lines=True)
            if not df.empty:
                # Expect columns: node, neighbour, weight
                if df.shape[1] >= 3:
                    df = df.iloc[:, :3]
                    df.columns = ['u', 'v', 'w']
                    df['w'] = pd.to_numeric(df['w'], errors='coerce').fillna(0.0)
                elif df.shape[1] >= 2:
                    # Fallback: treat as unweighted with weight = 1.0
                    df = df.iloc[:, :2]
                    df.columns = ['u', 'v']
                    df['w'] = 1.0
                else:
                    rows.append(tuple(metrics.values()))
                    continue
                
                su = df.groupby('u')['w'].sum()
                sv = df.groupby('v')['w'].sum()
                strength = su.add(sv, fill_value=0.0)
                n = len(strength)
                if n > 0:
                    # Only update if advanced metrics weren't available
                    if metrics['avg_degree'] == 0.0:
                        metrics['avg_degree'] = float(strength.sum() / n)
                    if metrics['max_degree'] == 0.0:
                        metrics['max_degree'] = float(strength.max())
                        
            rows.append(tuple(metrics.values()))
            
        out_path = os.path.join(folder, output_name)
        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(list(metrics.keys()))
            writer.writerows(rows)

def summarizeUnweightedDegrees(x, y, model='linard', output_name='unweighted_degree_stats.csv'):
    for x0, y0 in zip(x, y):
        folder = f"data/{model}/{x0}_{y0}"
        # Exclude _metrics.csv files from the graph files list
        files = sorted([f for f in glob(os.path.join(folder, "unweighted_graph_*.csv")) 
                       if not f.endswith('_metrics.csv')])
        rows = []
        print(folder)
        for fp in files:
            m = re.search(r'unweighted_graph_(.+)\.csv$', os.path.basename(fp))
            param = m.group(1) if m else os.path.basename(fp)
            
            # Initialize metrics with default values
            metrics = {
                'param': param,
                'avg_degree': 0.0,
                'max_degree': 0.0,
                'degree_entropy': 0.0,
                'avg_path_length': 0.0,
                'clustering_coefficient': 0.0,
                'density': 0.0,
                'mean_degree': 0.0,
                'median_degree': 0.0,
                'min_degree': 0.0,
                'max_degree_stats': 0.0
            }
            
            # Read advanced metrics if available
            metrics_file = fp.replace('.csv', '_metrics.csv')
            if os.path.exists(metrics_file):
                try:
                    metrics_df = pd.read_csv(metrics_file)
                    for _, row in metrics_df.iterrows():
                        metric_name = row['metric']
                        if metric_name in metrics:
                            metrics[metric_name] = float(row['value'])
                        elif metric_name == 'max_degree':
                            metrics['max_degree_stats'] = float(row['value'])
                except Exception as e:
                    print(f"Warning: Could not read metrics from {metrics_file}: {e}")
            
            # Calculate basic degree statistics as fallback
            df = pd.read_csv(fp, header=None, comment='#', skip_blank_lines=True)
            if not df.empty:
                # Expect columns: node, neighbour, weight
                if df.shape[1] >= 3:
                    df = df.iloc[:, :3]
                    df.columns = ['u', 'v', 'w']
                    df['w'] = pd.to_numeric(df['w'], errors='coerce').fillna(0.0)
                elif df.shape[1] >= 2:
                    # Treat as unweighted (degree count)
                    df = df.iloc[:, :2]
                    df.columns = ['u', 'v']
                    df['w'] = 1.0
                else:
                    rows.append(tuple(metrics.values()))
                    continue
                
                su = df.groupby('u')['w'].sum()
                sv = df.groupby('v')['w'].sum()
                strength = su.add(sv, fill_value=0.0)
                n = len(strength)
                if n > 0:
                    # Only update if advanced metrics weren't available
                    if metrics['avg_degree'] == 0.0:
                        metrics['avg_degree'] = float(strength.sum() / n)
                    if metrics['max_degree'] == 0.0:
                        metrics['max_degree'] = float(strength.max())
                        
            rows.append(tuple(metrics.values()))
            
        out_path = os.path.join(folder, output_name)
        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(list(metrics.keys()))
            writer.writerows(rows)