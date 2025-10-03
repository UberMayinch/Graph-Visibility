import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import csv
from glob import glob

def constructGraphs(x, y, model='linard'):
    for x0, y0 in zip(x, y):
        subprocess.run(f"./swap_uv.sh data/{model}/{x0}_{y0}", shell=True, check=True)
        weighted_command = f"./unweighted_construct data/{model}/{x0}_{y0}/"
        unweighted_command = f"./weighted_construct data/{model}/{x0}_{y0}/"
        subprocess.run(unweighted_command, shell=True, check=True)
        subprocess.run(weighted_command, shell=True, check=True)
        subprocess.run(f"./swap_uv.sh data/{model}/{x0}_{y0}", shell=True, check=True)

def summarizeWeightedDegrees(x, y, model='linard', output_name='weighted_degree_stats.csv'):
    for x0, y0 in zip(x, y):
        folder = f"data/{model}/{x0}_{y0}"
        files = sorted(glob(os.path.join(folder, "weighted_graph_*.csv")))
        rows = []
        print(folder)
        for fp in files:
            m = re.search(r'weighted_graph_(.+)\.csv$', os.path.basename(fp))
            param = m.group(1) if m else os.path.basename(fp)
            df = pd.read_csv(fp, header=None, comment='#', skip_blank_lines=True)
            if df.empty:
                avg_degree = 0.0
                max_degree = 0.0
            else:
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
                    avg_degree = 0.0
                    max_degree = 0.0
                    rows.append((param, avg_degree, max_degree))
                    continue
                su = df.groupby('u')['w'].sum()
                sv = df.groupby('v')['w'].sum()
                strength = su.add(sv, fill_value=0.0)
                n = len(strength)
                if n == 0:
                    avg_degree = 0.0
                    max_degree = 0.0
                else:
                    avg_degree = float(strength.sum() / n)
                    max_degree = float(strength.max())
            rows.append((param, avg_degree, max_degree))
        out_path = os.path.join(folder, output_name)
        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['param', 'avg_degree', 'max_degree'])
            writer.writerows(rows)

def summarizeUnweightedDegrees(x, y, model='linard', output_name='weighted_degree_stats.csv'):
    for x0, y0 in zip(x, y):
        folder = f"data/{model}/{x0}_{y0}"
        files = sorted(glob(os.path.join(folder, "unweighted_graph_*.csv")))
        rows = []
        print(folder)
        for fp in files:
            m = re.search(r'weighted_graph_(.+)\.csv$', os.path.basename(fp))
            param = m.group(1) if m else os.path.basename(fp)
            df = pd.read_csv(fp, header=None, comment='#', skip_blank_lines=True)
            if df.empty:
                avg_degree = 0.0
                max_degree = 0.0
            else:
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
                    avg_degree = 0.0
                    max_degree = 0.0
                    rows.append((param, avg_degree, max_degree))
                    continue
                su = df.groupby('u')['w'].sum()
                sv = df.groupby('v')['w'].sum()
                strength = su.add(sv, fill_value=0.0)
                n = len(strength)
                if n == 0:
                    avg_degree = 0.0
                    max_degree = 0.0
                else:
                    avg_degree = float(strength.sum() / n)
                    max_degree = float(strength.max())
            rows.append((param, avg_degree, max_degree))
        out_path = os.path.join(folder, output_name)
        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['param', 'avg_degree', 'max_degree'])
            writer.writerows(rows)