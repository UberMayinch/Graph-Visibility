import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import csv
import struct
import numpy as np
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
    # Exclude _metrics.bin files from the graph files list  
    files = sorted([f for f in glob(os.path.join(folder, "weighted_graph_*.bin")) 
                   if not f.endswith('_metrics.bin')])
    rows = []
    
    for fp in files:
        m = re.search(r'weighted_graph_(.+)\.bin$', os.path.basename(fp))
        param_str = m.group(1) if m else os.path.basename(fp)
        try:
            param = float(param_str)
        except Exception:
            import numpy as _np
            param = _np.nan
        
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
        metrics_file = fp.replace('.bin', '_metrics.bin')
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'rb') as mf:
                    if mf.read(4) != b'MET1':
                        raise ValueError('Invalid MET1 magic')
                    n = struct.unpack('<I', mf.read(4))[0]
                    md = {}
                    for _ in range(n):
                        klen = struct.unpack('<H', mf.read(2))[0]
                        key = mf.read(klen).decode('ascii')
                        val = struct.unpack('<d', mf.read(8))[0]
                        md[key] = val
                for metric_name in metrics.keys():
                    if metric_name in md:
                        metrics[metric_name] = float(md[metric_name])
                    elif metric_name == 'max_degree_stats' and 'max_degree' in md:
                        metrics['max_degree_stats'] = float(md['max_degree'])
            except Exception as e:
                print(f"Warning: Could not read metrics from {metrics_file}: {e}")
        
        # Calculate basic degree statistics as fallback (optimized)
        try:
            # Read WGB1
            with open(fp, 'rb') as gf:
                if gf.read(4) != b'WGB1':
                    raise ValueError('Invalid WGB1 magic')
                m = struct.unpack('<Q', gf.read(8))[0]
                # Accumulate node strengths from edges
                strength = {}
                for _ in range(m):
                    u = struct.unpack('<i', gf.read(4))[0]
                    v = struct.unpack('<i', gf.read(4))[0]
                    w = struct.unpack('<d', gf.read(8))[0]
                    strength[u] = strength.get(u, 0.0) + w
                    strength[v] = strength.get(v, 0.0) + w
                if strength:
                    n_nodes = len(strength)
                    if metrics['avg_degree'] == 0.0:
                        metrics['avg_degree'] = float(sum(strength.values()) / n_nodes)
                    if metrics['max_degree'] == 0.0:
                        metrics['max_degree'] = float(max(strength.values()))
        except Exception as e:
            print(f"Warning: Error processing {fp}: {e}")
                    
        rows.append(tuple(metrics.values()))
        
    return folder, rows, list(metrics.keys())

def summarizeWeightedDegrees(x, y, model='linard', output_name='weighted_degree_stats.csv'):
    """Process weighted degree summary sequentially (optimized; binary pipeline)."""
    for x0, y0 in zip(x, y):
        folder = f"data/{model}/{x0}_{y0}"
        print(f"Processing {folder}")
        # Collect weighted graph .bin files excluding metrics
        files = sorted([f for f in glob(os.path.join(folder, "weighted_graph_*.bin"))
                        if not f.endswith('_metrics.bin')])
        rows = []
        for fp in files:
            m = re.search(r'weighted_graph_(.+)\.bin$', os.path.basename(fp))
            param_str = m.group(1) if m else os.path.basename(fp)
            try:
                param = float(param_str)
            except Exception:
                import numpy as _np
                param = _np.nan
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
            # Prefer MET1 metrics
            metrics_file = fp.replace('.bin', '_metrics.bin')
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'rb') as mf:
                        if mf.read(4) != b'MET1':
                            raise ValueError('Invalid MET1 magic')
                        n = struct.unpack('<I', mf.read(4))[0]
                        md = {}
                        for _ in range(int(n)):
                            klen = struct.unpack('<H', mf.read(2))[0]
                            key = mf.read(int(klen)).decode('ascii')
                            val = struct.unpack('<d', mf.read(8))[0]
                            md[key] = val
                    for k in list(metrics.keys()):
                        if k in md:
                            metrics[k] = float(md[k])
                        elif k == 'max_degree_stats' and 'max_degree' in md:
                            metrics['max_degree_stats'] = float(md['max_degree'])
                except Exception as e:
                    print(f"Warning: Could not read metrics from {metrics_file}: {e}")
            # Fallback: compute simple stats from WGB1
            if metrics['avg_degree'] == 0.0 or metrics['max_degree'] == 0.0:
                try:
                    with open(fp, 'rb') as gf:
                        if gf.read(4) != b'WGB1':
                            raise ValueError('Invalid WGB1 magic')
                        mcount = struct.unpack('<Q', gf.read(8))[0]
                        strength = {}
                        for _ in range(int(mcount)):
                            u = struct.unpack('<i', gf.read(4))[0]
                            v = struct.unpack('<i', gf.read(4))[0]
                            w = struct.unpack('<d', gf.read(8))[0]
                            strength[u] = strength.get(u, 0.0) + w
                            strength[v] = strength.get(v, 0.0) + w
                        if strength:
                            n_nodes = len(strength)
                            if metrics['avg_degree'] == 0.0:
                                metrics['avg_degree'] = float(sum(strength.values()) / n_nodes)
                            if metrics['max_degree'] == 0.0:
                                metrics['max_degree'] = float(max(strength.values()))
                except Exception as e:
                    print(f"Warning: Error processing {fp}: {e}")
            rows.append(tuple(metrics.values()))
        if rows:
            out_path = os.path.join(folder, output_name.replace('.csv', '.bin'))
            cols = list(metrics.keys())
            with open(out_path, 'wb') as f:
                f.write(b'STB1')
                f.write(struct.pack('<I', len(cols)))
                for c in cols:
                    b = c.encode('ascii')
                    f.write(struct.pack('<H', len(b)))
                    f.write(b)
                f.write(struct.pack('<I', len(rows)))
                arr = np.array(rows, dtype=np.float64)
                f.write(arr.tobytes(order='C'))
            print(f"Weighted degree statistics saved to {out_path}")

def summarizeUnweightedDegrees(x, y, model='linard', output_name='unweighted_degree_stats.csv'):
    """Process unweighted degree summary sequentially (optimized but simple)."""
    
    for x0, y0 in zip(x, y):
        folder = f"data/{model}/{x0}_{y0}"
        print(f"Processing {folder}")
        
        # Exclude _metrics.bin files from the graph files list
        files = sorted([f for f in glob(os.path.join(folder, "unweighted_graph_*.bin")) 
                       if not f.endswith('_metrics.bin')])
        rows = []
        for fp in files:
            m = re.search(r'unweighted_graph_(.+)\.bin$', os.path.basename(fp))
            param_str = m.group(1) if m else os.path.basename(fp)
            try:
                param = float(param_str)
            except Exception:
                import numpy as _np
                param = _np.nan
            
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
            metrics_file = fp.replace('.bin', '_metrics.bin')
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'rb') as mf:
                        if mf.read(4) != b'MET1':
                            raise ValueError('Invalid MET1 magic')
                        n = struct.unpack('<I', mf.read(4))[0]
                        md = {}
                        for _ in range(n):
                            klen = struct.unpack('<H', mf.read(2))[0]
                            key = mf.read(klen).decode('ascii')
                            val = struct.unpack('<d', mf.read(8))[0]
                            md[key] = val
                    for metric_name in metrics.keys():
                        if metric_name in md:
                            metrics[metric_name] = float(md[metric_name])
                        elif metric_name == 'max_degree_stats' and 'max_degree' in md:
                            metrics['max_degree_stats'] = float(md['max_degree'])
                except Exception as e:
                    print(f"Warning: Could not read metrics from {metrics_file}: {e}")
            
            # Calculate basic degree statistics as fallback
            try:
                # Read UGB1
                with open(fp, 'rb') as gf:
                    if gf.read(4) != b'UGB1':
                        raise ValueError('Invalid UGB1 magic')
                    m = struct.unpack('<Q', gf.read(8))[0]
                    degree = {}
                    for _ in range(m):
                        u = struct.unpack('<i', gf.read(4))[0]
                        v = struct.unpack('<i', gf.read(4))[0]
                        degree[u] = degree.get(u, 0) + 1
                        degree[v] = degree.get(v, 0) + 1
                    if degree:
                        n_nodes = len(degree)
                        if metrics['avg_degree'] == 0.0:
                            metrics['avg_degree'] = float(sum(degree.values()) / n_nodes)
                        if metrics['max_degree'] == 0.0:
                            metrics['max_degree'] = float(max(degree.values()))
            except Exception as e:
                print(f"Warning: Error processing {fp}: {e}")
                        
            rows.append(tuple(metrics.values()))
            
        # Write results for this folder
        if rows:
            out_path = os.path.join(folder, output_name.replace('.csv', '.bin'))
            cols = list(metrics.keys())
            with open(out_path, 'wb') as f:
                f.write(b'STB1')
                f.write(struct.pack('<I', len(cols)))
                for c in cols:
                    b = c.encode('ascii')
                    f.write(struct.pack('<H', len(b)))
                    f.write(b)
                f.write(struct.pack('<I', len(rows)))
                arr = np.array(rows, dtype=np.float64)
                f.write(arr.tobytes(order='C'))
            print(f"Unweighted degree statistics saved to {out_path}")