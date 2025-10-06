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
    """Construct graphs for all initial conditions (kept for backward compatibility)."""
    from concurrent.futures import ThreadPoolExecutor
    import multiprocessing as mp
    
    def construct_single(x0_y0):
        x0, y0 = x0_y0
        folder_path = f"data/{model}/{x0}_{y0}"
        
        try:
            # Construct graphs (optimized C++ code with parallel processing)
            weighted_command = f"./weighted_construct {folder_path}/"
            unweighted_command = f"./unweighted_construct {folder_path}/"
            subprocess.run(weighted_command, shell=True, check=True, capture_output=True)
            subprocess.run(unweighted_command, shell=True, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    # Run in parallel
    tasks = list(zip(x, y))
    max_workers = min(mp.cpu_count(), len(tasks))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(construct_single, tasks))
    
    success_count = sum(results)
    print(f"Graph construction: {success_count}/{len(tasks)} successful")

def _process_single_folder_weighted(args):
    """Process a single folder for weighted degrees (separate function for pickling).""" 
    import numpy as np
    x0, y0, model = args
    folder = f"data/{model}/{x0}_{y0}"
    # Exclude _metrics.csv files from the graph files list  
    files = sorted([f for f in glob(os.path.join(folder, "weighted_graph_*.csv")) 
                   if not f.endswith('_metrics.csv')])
    rows = []
    
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
        
        # Read advanced metrics if available (optimized)
        metrics_file = fp.replace('.csv', '_metrics.csv')
        if os.path.exists(metrics_file):
            try:
                # Use faster CSV reading with numpy
                metrics_df = pd.read_csv(metrics_file, engine='c')
                metrics_dict = dict(zip(metrics_df['metric'], metrics_df['value']))
                for metric_name in metrics.keys():
                    if metric_name in metrics_dict:
                        metrics[metric_name] = float(metrics_dict[metric_name])
                    elif metric_name == 'max_degree_stats' and 'max_degree' in metrics_dict:
                        metrics['max_degree_stats'] = float(metrics_dict['max_degree'])
            except Exception as e:
                print(f"Warning: Could not read metrics from {metrics_file}: {e}")
        
        # Calculate basic degree statistics as fallback (optimized)
        try:
            # Fast CSV reading with optimized settings
            df = pd.read_csv(fp, header=None, comment='#', skip_blank_lines=True, 
                           engine='c', dtype=str)
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
        except Exception as e:
            print(f"Warning: Error processing {fp}: {e}")
                    
        rows.append(tuple(metrics.values()))
        
    return folder, rows, list(metrics.keys())

def summarizeWeightedDegrees(x, y, model='linard', output_name='weighted_degree_stats.csv'):
    """Process weighted degree summary sequentially (optimized but simple)."""
    
    for x0, y0 in zip(x, y):
        folder = f"data/{model}/{x0}_{y0}"
        print(f"Processing {folder}")
        
        # Exclude _metrics.csv files from the graph files list  
        files = sorted([f for f in glob(os.path.join(folder, "weighted_graph_*.csv")) 
                       if not f.endswith('_metrics.csv')])
        rows = []
        
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
                    metrics_df = pd.read_csv(metrics_file, engine='c')
                    metrics_dict = dict(zip(metrics_df['metric'], metrics_df['value']))
                    for metric_name in metrics.keys():
                        if metric_name in metrics_dict:
                            metrics[metric_name] = float(metrics_dict[metric_name])
                        elif metric_name == 'max_degree_stats' and 'max_degree' in metrics_dict:
                            metrics['max_degree_stats'] = float(metrics_dict['max_degree'])
                except Exception as e:
                    print(f"Warning: Could not read metrics from {metrics_file}: {e}")
            
            # Calculate basic degree statistics as fallback
            try:
                df = pd.read_csv(fp, header=None, comment='#', skip_blank_lines=True, 
                               engine='c', dtype=str)
                if not df.empty:
                    if df.shape[1] >= 3:
                        df = df.iloc[:, :3]
                        df.columns = ['u', 'v', 'w']
                        df['w'] = pd.to_numeric(df['w'], errors='coerce').fillna(0.0)
                    elif df.shape[1] >= 2:
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
                        if metrics['avg_degree'] == 0.0:
                            metrics['avg_degree'] = float(strength.sum() / n)
                        if metrics['max_degree'] == 0.0:
                            metrics['max_degree'] = float(strength.max())
            except Exception as e:
                print(f"Warning: Error processing {fp}: {e}")
                        
            rows.append(tuple(metrics.values()))
            
        # Write results for this folder
        if rows:
            out_path = os.path.join(folder, output_name)
            with open(out_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(list(metrics.keys()))
                writer.writerows(rows)
            print(f"Weighted degree statistics saved to {out_path}")

def summarizeUnweightedDegrees(x, y, model='linard', output_name='unweighted_degree_stats.csv'):
    """Process unweighted degree summary sequentially (optimized but simple)."""
    
    for x0, y0 in zip(x, y):
        folder = f"data/{model}/{x0}_{y0}"
        print(f"Processing {folder}")
        
        # Exclude _metrics.csv files from the graph files list
        files = sorted([f for f in glob(os.path.join(folder, "unweighted_graph_*.csv")) 
                       if not f.endswith('_metrics.csv')])
        rows = []
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
                    metrics_df = pd.read_csv(metrics_file, engine='c')
                    metrics_dict = dict(zip(metrics_df['metric'], metrics_df['value']))
                    for metric_name in metrics.keys():
                        if metric_name in metrics_dict:
                            metrics[metric_name] = float(metrics_dict[metric_name])
                        elif metric_name == 'max_degree_stats' and 'max_degree' in metrics_dict:
                            metrics['max_degree_stats'] = float(metrics_dict['max_degree'])
                except Exception as e:
                    print(f"Warning: Could not read metrics from {metrics_file}: {e}")
            
            # Calculate basic degree statistics as fallback
            try:
                df = pd.read_csv(fp, header=None, comment='#', skip_blank_lines=True, 
                               engine='c', dtype=str)
                if not df.empty:
                    # Expect columns: node, neighbour for unweighted graphs
                    if df.shape[1] >= 2:
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
                        if metrics['avg_degree'] == 0.0:
                            metrics['avg_degree'] = float(strength.sum() / n)
                        if metrics['max_degree'] == 0.0:
                            metrics['max_degree'] = float(strength.max())
            except Exception as e:
                print(f"Warning: Error processing {fp}: {e}")
                        
            rows.append(tuple(metrics.values()))
            
        # Write results for this folder
        if rows:
            out_path = os.path.join(folder, output_name)
            with open(out_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(list(metrics.keys()))
                writer.writerows(rows)
            print(f"Unweighted degree statistics saved to {out_path}")