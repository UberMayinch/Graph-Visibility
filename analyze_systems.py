#!/usr/bin/env python3
"""
Chaotic Systems Visibility Graph Analysis Script

This script performs comprehensive analysis of chaotic dynamical systems using visibility graphs.
It supports multiple models (FHN, Linard) with configurable parameters and initial conditions.

Usage:
    python analyze_systems.py [--config config.json] [--model fhn|linard|all] [--verbose]

Features:
    - JSON-based configuration management
    - Modular analysis pipeline
    - Time series processing and visualization
    - Bifurcation diagram generation  
    - Visibility graph construction and analysis
    - Degree distribution analysis
    - Metric envelope plotting across initial conditions
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
import zipfile
from glob import glob
import struct

# Add pyutils to path for imports
sys.path.insert(0, 'pyutils')

from pyutils.appendlocal import AppendLocalOptima
from pyutils.processts import process_timeseries
from pyutils.bifurcation import bifurcationdiagram, bifurcation_smoothed
from pyutils.graphmetrics import constructGraphs, summarizeWeightedDegrees, summarizeUnweightedDegrees


class SystemAnalyzer:
    """Main class for analyzing chaotic systems with visibility graphs."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize analyzer with configuration."""
        self.config = self._load_config(config_path)
        self._setup_environment()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"✓ Loaded configuration from {config_path}")
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    def _setup_environment(self):
        """Set up random seeds and compile executables."""
        # Set random seeds for reproducibility
        seed = self.config['general']['random_seed']
        np.random.seed(seed)
        random.seed(seed)
        print(f"✓ Set random seed to {seed}")
        
        # Compile executables
        self._compile_executables()
        
        # Create directories
        self._create_directories()
    
    def _compile_executables(self):
        """Compile C++ executables using Makefile."""
        try:
            print("📦 Compiling executables...")
            subprocess.run("make all", shell=True, check=True, capture_output=True)
            print("✓ Executables compiled successfully")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to compile executables: {e}")
    
    def _create_directories(self):
        """Create necessary directories."""
        try:
            print("📁 Creating directories...")
            subprocess.run("make dirs", shell=True, check=True, capture_output=True)
            print("✓ Directories created successfully")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create directories: {e}")
    
    def generate_initial_conditions(self, model: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate initial conditions for specified model."""
        model_config = self.config[model]['initial_conditions']
        precision = self.config['general']['precision_degree']
        
        if model == 'fhn':
            u_count = model_config['u_count']
            v_count = model_config['v_count']
            u_base = model_config['u_base']
            u_range = model_config['u_range']
            v_base = model_config['v_base']
            v_range = model_config['v_range']
            
            u = np.random.random(size=u_count) * u_range + u_base
            v = np.random.random(size=v_count) * v_range + v_base
            u = np.round(u, precision)
            v = np.round(v, precision)
            
            print(f"Generated FHN initial conditions:")
            print(f"  u values: {u}")
            print(f"  v values: {v}")
            
            return u, v
            
        elif model == 'linard':
            x_count = model_config['x_count']
            y_count = model_config['y_count']
            x_base = model_config['x_base']
            x_range = model_config['x_range']
            y_base = model_config['y_base']
            y_range = model_config['y_range']
            
            x = np.random.random(size=x_count) * x_range + x_base - x_range/2
            y = np.random.random(size=y_count) * y_range + y_base
            x = np.round(x, precision)
            y = np.round(y, precision)
            
            print(f"Generated Linard initial conditions:")
            print(f"  x values: {x}")
            print(f"  y values: {y}")
            
            return x, y
        
        else:
            raise ValueError(f"Unknown model: {model}")
    
    def generate_parameter_range(self, model: str) -> np.ndarray:
        """Generate parameter range for specified model."""
        model_config = self.config[model]['parameters']
        num_params = self.config['general']['num_params']
        precision = self.config['general']['precision_degree']
        
        if model == 'fhn':
            A_min = model_config['A_min']
            A_max = model_config['A_max']
            A = np.round(np.linspace(A_min, A_max, num_params), precision)
            print(f"Generated FHN parameter range A: [{A_min}, {A_max}] with {num_params} points")
            return A
            
        elif model == 'linard':
            omega_min = model_config['omega_min']
            omega_max = model_config['omega_max']
            omega = np.round(np.linspace(omega_min, omega_max, num_params), precision)
            print(f"Generated Linard parameter range omega: [{omega_min}, {omega_max}] with {num_params} points")
            return omega
        
        else:
            raise ValueError(f"Unknown model: {model}")
    
    def _run_single_simulation(self, args):
        """Run a single simulation (sequential wrapper)."""
        model, param, ic1_val, ic2_val, num_steps = args
        
        # Create directory for this initial condition
        make_dir_cmd = f"mkdir -p -- data/{model}/{ic1_val}_{ic2_val}"
        subprocess.run(make_dir_cmd, shell=True)
        
        # Run simulation
        cmd = f"./{model} {param} {ic1_val} {ic2_val} {num_steps}"
        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
            return (ic1_val, ic2_val, param, f"data/{model}/{ic1_val}_{ic2_val}/output_{param}.bin")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Warning: Simulation failed for {model} with params {param}, {ic1_val}, {ic2_val}")
            return None
    
    def run_simulations(self, model: str, initial_conditions: Tuple[np.ndarray, np.ndarray], 
                       parameters: np.ndarray) -> Dict[Tuple[float, float], Dict[float, str]]:
        """Run simulations sequentially for all parameter values and initial conditions."""
        ic1, ic2 = initial_conditions
        csv_paths: Dict[Tuple[float, float], Dict[float, str]] = defaultdict(dict)
        
        print(f"🚀 Running {model.upper()} simulations sequentially...")
        
        # Get number of steps from configuration
        num_steps = self.config['simulation'][f'{model}_num_steps']
        
        # Prepare all simulation tasks
        simulation_tasks: List[Tuple[str, float, float, float, int]] = []
        for ic1_val, ic2_val in zip(ic1, ic2):
            for param in parameters:
                simulation_tasks.append((model, float(param), float(ic1_val), float(ic2_val), int(num_steps)))
        
        # Run simulations sequentially
        completed = 0
        total = len(simulation_tasks)
        for task in simulation_tasks:
            result = self._run_single_simulation(task)
            if result is not None:
                ic1_val, ic2_val, param, path = result
                csv_paths[(ic1_val, ic2_val)][param] = path
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"Progress: {completed}/{total} simulations completed")
        
        print(f"✓ {model.upper()} simulations completed")
        return csv_paths
    
    def load_dataframes(self, model: str, csv_paths: Dict[Tuple[float, float], Dict[float, str]], 
                       column_name: str) -> Dict[Tuple[float, float], Dict[float, pd.DataFrame]]:
        """Load and process dataframes with local optima."""
        dataframes = defaultdict(dict)
        stabilizing_time = self.config['general']['stabilizing_time']
        window_size = self.config['general']['window_size']
        
        print(f"📊 Loading {model.upper()} dataframes...")
        
        for (ic1, ic2), param_paths in csv_paths.items():
            for param, path in param_paths.items():
                try:
                    dataframes[(ic1, ic2)][param] = AppendLocalOptima(
                        path, stabilizing_time, window_size, column_name
                    )
                except Exception as e:
                    print(f"⚠️  Warning: Failed to load {path}: {e}")
                    continue
        
        print(f"✓ {model.upper()} dataframes loaded")
        return dataframes
    
    def process_time_series(self, model: str, initial_conditions: Tuple[np.ndarray, np.ndarray],
                           dataframes: Dict[Tuple[float, float], Dict[float, pd.DataFrame]],
                           parameters: np.ndarray):
        """Process time series for all initial conditions."""
        if not self.config[model]['analysis']['process_timeseries']:
            print(f"⏭️  Skipping time series processing for {model.upper()}")
            return
        
        print(f"🔄 Processing {model.upper()} time series...")
        
        ic1, ic2 = initial_conditions
        model_config = self.config[model]['parameters']
        
        if model == 'fhn':
            col1, col2 = model_config['column_u'], model_config['column_v']
        elif model == 'linard':
            col1, col2 = model_config['column_x'], model_config['column_y']
        
        for (ic1_val, ic2_val) in zip(ic1, ic2):
            if (ic1_val, ic2_val) not in dataframes:
                continue
                
            # Sort dataframes by parameter values
            sorted_items = sorted(dataframes[(ic1_val, ic2_val)].items())
            dataframes_list = [df for param, df in sorted_items]
            
            print(f"  Processing {len(dataframes_list)} dataframes for IC ({ic1_val:.6f}, {ic2_val:.6f})")
            
            # Create plots directory
            make_dir_cmd = f"mkdir -p -- plots/{model}/{ic1_val}_{ic2_val}"
            subprocess.run(make_dir_cmd, shell=True)
            
            # Process time series
            process_timeseries(parameters, dataframes_list, col1, col2, model, 
                             mode='all', initial_conds=f"./{ic1_val}_{ic2_val}")
        
        print(f"✓ {model.upper()} time series processing completed")
    
    def generate_bifurcation_diagrams(self, model: str, initial_conditions: Tuple[np.ndarray, np.ndarray],
                                    dataframes: Dict[Tuple[float, float], Dict[float, pd.DataFrame]]):
        """Generate bifurcation diagrams."""
        if not self.config[model]['analysis']['bifurcation_diagram']:
            print(f"⏭️  Skipping bifurcation diagrams for {model.upper()}")
            return
        
        print(f"📈 Generating {model.upper()} bifurcation diagrams...")
        
        ic1, ic2 = initial_conditions
        model_config = self.config[model]['parameters']
        
        if model == 'fhn':
            col = model_config['column_u']
        elif model == 'linard':
            col = model_config['column_x']
        
        peak_data = {}
        
        for (ic1_val, ic2_val) in zip(ic1, ic2):
            if (ic1_val, ic2_val) not in dataframes:
                continue
                
            print(f"  Generating bifurcation diagram for IC ({ic1_val:.6f}, {ic2_val:.6f})")
            peak_data[ic1_val, ic2_val] = bifurcationdiagram(
                dataframes[(ic1_val, ic2_val)], 
                col=col,
                initial_conds=f"./{ic1_val}_{ic2_val}", 
                model=model
            )
        
        # Generate smoothed bifurcation diagram for FHN
        if model == 'fhn' and peak_data:
            bifurcation_smoothed(peak_data)
        
        print(f"✓ {model.upper()} bifurcation diagrams completed")
    
    def _construct_single_ic_graphs(self, args):
        """Construct graphs for a single initial condition (for parallel execution)."""
        x0, y0, model = args
        folder_path = f"data/{model}/{x0}_{y0}"
        
        try:
            # Construct graphs using optimized C++ utilities
            weighted_command = f"./weighted_construct {folder_path}/"
            unweighted_command = f"./unweighted_construct {folder_path}/"
            
            subprocess.run(weighted_command, shell=True, check=True, capture_output=True)
            subprocess.run(unweighted_command, shell=True, check=True, capture_output=True)
            return (x0, y0, True)
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Warning: Graph construction failed for {model} IC ({x0}, {y0})")
            return (x0, y0, False)
    
    def construct_visibility_graphs(self, model: str, initial_conditions: Tuple[np.ndarray, np.ndarray]):
        """Construct visibility graphs sequentially per IC and finalize per folder.

        For each initial condition (IC) folder:
          - construct weighted/unweighted graphs (sequential)
          - compute metrics for graphs within that folder (parallel threads)
          - optionally zip & prune that folder if enabled in config
        """
        if not self.config[model]['analysis']['construct_graphs']:
            print(f"⏭️  Skipping graph construction for {model.upper()}")
            return
        
        print(f"🔧 Constructing {model.upper()} visibility graphs sequentially...")
        
        ic1, ic2 = initial_conditions
        
        # Process per IC: construct -> metrics (per folder) -> optional zip/prune
        tasks: List[Tuple[float, float, str]] = [(float(x0), float(y0), model) for x0, y0 in zip(ic1, ic2)]
        completed = 0
        total = len(tasks)
        for task in tasks:
            x0, y0, _ = task
            # 1) Construct graphs for this IC
            _, _, success = self._construct_single_ic_graphs(task)
            if not success:
                print(f"⚠️  Warning: Graph construction failed for {model} IC ({x0}, {y0})")
                completed += 1
                continue
            # 2) Compute metrics per-folder if not skipped
            if self.config.get('execution_mode', {}).get('calculate_advanced_metrics', True) \
               and os.getenv('GV_SKIP_METRICS', '0') != '1':
                self._calculate_graph_metrics_for_ic(model, x0, y0, workers=self._get_metrics_workers())
            # 3) Optionally zip & prune this IC folder
            self._zip_and_prune_single_ic(model, x0, y0)
            completed += 1
            if completed % 5 == 0 or completed == total:
                print(f"Progress: {completed}/{total} ICs finalized")
        
        print(f"✓ {model.upper()} visibility graphs completed")
    
    def _calculate_single_graph_metrics(self, args):
        """Calculate metrics for a single graph file (for parallel execution)."""
        graph_path, metrics_file = args
        
        try:
            cmd = f"./graph_metrics {graph_path} {metrics_file}"
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
            return (graph_path, True)
        except subprocess.CalledProcessError as e:
            return (graph_path, False)
    
    def _calculate_advanced_graph_metrics(self, model: str, initial_conditions: Tuple[np.ndarray, np.ndarray]):
        """Calculate advanced graph metrics using C++ utilities in parallel (per-file)."""
        # Determine concurrency: env METRICS_JOBS > SLURM_CPUS_PER_TASK > config > cpu_count
        env_jobs = os.getenv('METRICS_JOBS')
        slurm_cpus = os.getenv('SLURM_CPUS_PER_TASK')
        cfg_jobs = self.config.get('execution_mode', {}).get('metrics_workers')
        try:
            workers = int(env_jobs) if env_jobs and int(env_jobs) > 0 else None
        except Exception:
            workers = None
        if workers is None:
            try:
                workers = int(slurm_cpus) if slurm_cpus and int(slurm_cpus) > 0 else None
            except Exception:
                workers = None
        if workers is None and isinstance(cfg_jobs, int) and cfg_jobs > 0:
            workers = cfg_jobs
        if workers is None:
            workers = max(1, (os.cpu_count() or 1))

        print(f"📈 Calculating advanced graph metrics for {model.upper()} with {workers} workers...")
        
        ic1, ic2 = initial_conditions
        
        # Collect all graph files that need metrics calculation
        tasks = []
        for ic1_val, ic2_val in zip(ic1, ic2):
            ic_dir = f"data/{model}/{ic1_val}_{ic2_val}"
            if not os.path.exists(ic_dir):
                continue
            
            # Find all graph files in the directory (excluding existing metrics files)
            graph_files = [f for f in os.listdir(ic_dir) 
                          if 'graph' in f and f.endswith('.bin') and not f.endswith('_metrics.bin')]
            
            for graph_file in graph_files:
                graph_path = os.path.join(ic_dir, graph_file)
                metrics_file = graph_path.replace('.bin', '_metrics.bin')
                tasks.append((graph_path, metrics_file))
        
        if not tasks:
            print("No graph files found for metrics calculation")
            return
        
        # Run metrics calculation in parallel (thread pool; subprocess calls are I/O bound)
        completed = 0
        success_count = 0
        total = len(tasks)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        # Ensure C++ stays single-threaded to avoid oversubscription
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._calculate_single_graph_metrics, t): t for t in tasks}
            for fut in as_completed(futures):
                try:
                    graph_path, success = fut.result()
                except Exception as e:
                    # Unexpected error in worker
                    t = futures[fut]
                    graph_path = t[0]
                    success = False
                if success:
                    success_count += 1
                else:
                    print(f"⚠️  Warning: Failed to calculate metrics for {os.path.basename(graph_path)}")
                completed += 1
                if completed % max(1, total // 10) == 0 or completed == total:
                    print(f"Progress: {completed}/{total} metrics calculations completed ({success_count} successful)")
        
        print(f"✓ {model.upper()} advanced graph metrics completed")

    def _get_metrics_workers(self) -> int:
        """Resolve worker count for metrics parallelism from env/Slurm/config."""
        env_jobs = os.getenv('METRICS_JOBS')
        slurm_cpus = os.getenv('SLURM_CPUS_PER_TASK')
        cfg_jobs = self.config.get('execution_mode', {}).get('metrics_workers')
        workers = None
        try:
            workers = int(env_jobs) if env_jobs and int(env_jobs) > 0 else None
        except Exception:
            workers = None
        if workers is None:
            try:
                workers = int(slurm_cpus) if slurm_cpus and int(slurm_cpus) > 0 else None
            except Exception:
                workers = None
        if workers is None and isinstance(cfg_jobs, int) and cfg_jobs > 0:
            workers = cfg_jobs
        if workers is None:
            workers = max(1, (os.cpu_count() or 1))
        return workers

    def _calculate_graph_metrics_for_ic(self, model: str, ic1_val: float, ic2_val: float, workers: Optional[int] = None):
        """Calculate metrics for all graphs within a single IC folder in parallel."""
        ic_dir = f"data/{model}/{ic1_val}_{ic2_val}"
        if not os.path.isdir(ic_dir):
            return
        graph_files = [f for f in os.listdir(ic_dir)
                       if 'graph' in f and f.endswith('.bin') and not f.endswith('_metrics.bin')]
        if not graph_files:
            return
        tasks = []
        for graph_file in graph_files:
            graph_path = os.path.join(ic_dir, graph_file)
            metrics_file = graph_path.replace('.bin', '_metrics.bin')
            tasks.append((graph_path, metrics_file))
        if not tasks:
            return
        if workers is None:
            workers = self._get_metrics_workers()
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self._calculate_single_graph_metrics, t) for t in tasks]
            for _ in as_completed(futures):
                pass

    def _zip_and_prune_single_ic(self, model: str, ic1_val: float, ic2_val: float):
        """Zip and prune artifacts for a single IC folder if enabled."""
        if not self.config.get('execution_mode', {}).get('zip_and_prune_enabled', False):
            return
        ic_dir = f"data/{model}/{ic1_val}_{ic2_val}"
        if not os.path.isdir(ic_dir):
            return
        timeseries_files = sorted(glob(os.path.join(ic_dir, 'output_*.bin')))
        weighted_files = sorted(glob(os.path.join(ic_dir, 'weighted_graph_*.bin')))
        unweighted_files = sorted(glob(os.path.join(ic_dir, 'unweighted_graph_*.bin')))
        categories = [
            (timeseries_files, f"timeseries_{ic1_val}_{ic2_val}.zip"),
            (weighted_files, f"weighted_graphs_{ic1_val}_{ic2_val}.zip"),
            (unweighted_files, f"unweighted_graphs_{ic1_val}_{ic2_val}.zip"),
        ]
        for files, zipname in categories:
            if not files:
                continue
            zip_path = os.path.join(ic_dir, zipname)
            with zipfile.ZipFile(zip_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
                for fp in files:
                    zf.write(fp, arcname=os.path.basename(fp))
            for fp in files:
                try:
                    os.remove(fp)
                except OSError:
                    pass
        print(f"Zipped and pruned IC folder {ic_dir}")
    
    def analyze_degree_distributions(self, model: str, initial_conditions: Tuple[np.ndarray, np.ndarray]):
        """Analyze weighted and unweighted degree distributions."""
        ic1, ic2 = initial_conditions
        
        # Weighted degree analysis
        if self.config[model]['analysis'].get('weighted_degree_summary', False):
            print(f"📊 Analyzing {model.upper()} weighted degree distributions...")
            output_file = self.config['output_files']['weighted_degree_stats']
            summarizeWeightedDegrees(ic1, ic2, model=model, output_name=output_file)
            print(f"✓ {model.upper()} weighted degree analysis completed")
        
        # Unweighted degree analysis
        if self.config[model]['analysis'].get('unweighted_degree_summary', False):
            print(f"📊 Analyzing {model.upper()} unweighted degree distributions...")
            output_file = self.config['output_files']['unweighted_degree_stats']
            summarizeUnweightedDegrees(ic1, ic2, model=model, output_name=output_file)
            print(f"✓ {model.upper()} unweighted degree analysis completed")
    
    def plot_metric_envelope_across_ics(self, model: str, metric_col: str, 
                                       stats_filename: str = "unweighted_degree_stats.bin",
                                       param_col: str = None, save: bool = True):
        """
        Plot metric envelopes across initial conditions.
        Aggregates the chosen metric across all initial condition folders and plots:
        - each realisation as a dotted, low-opacity line
        - the mean
        - +4σ and +6σ upper envelopes with different colors
        """
        base_dir = os.path.join("data", model)
        if not os.path.isdir(base_dir):
            raise FileNotFoundError(f"Base directory not found: {base_dir}")

        # Discover IC folders
        ic_dirs = [
            os.path.join(base_dir, d)
            for d in sorted(os.listdir(base_dir))
            if os.path.isdir(os.path.join(base_dir, d))
        ]
        if not ic_dirs:
            raise FileNotFoundError(f"No initial condition directories found under {base_dir}")

        # Try common parameter column names if not provided
        candidate_param_cols = ["A", "omega", "param", "parameter", "lambda"]
        if param_col is not None:
            candidate_param_cols = [param_col] + [c for c in candidate_param_cols if c != param_col]

        # Load per-IC metric series
        records = []
        used_param_col = None
        for ic_path in ic_dirs:
            stats_path = os.path.join(ic_path, stats_filename)
            if not os.path.isfile(stats_path):
                continue

            # Load STB1 (.bin) or CSV
            if stats_path.endswith('.bin'):
                try:
                    df = self._read_stb1(stats_path)
                except Exception as e:
                    print(f"⚠️  Warning: Failed to read STB1 table {stats_path}: {e}")
                    continue
            else:
                df = pd.read_csv(stats_path)
            # Infer parameter column
            chosen_param = None
            for cand in candidate_param_cols:
                if cand in df.columns:
                    chosen_param = cand
                    break
            if chosen_param is None:
                numeric_cols = [c for c in df.columns if c != metric_col and pd.api.types.is_numeric_dtype(df[c])]
                if not numeric_cols:
                    raise ValueError(f"Could not infer parameter column in {stats_path}")
                chosen_param = numeric_cols[0]

            if metric_col not in df.columns:
                raise ValueError(f"Metric column '{metric_col}' not found in {stats_path}. Available: {list(df.columns)}")

            sub = df[[chosen_param, metric_col]].dropna().rename(
                columns={chosen_param: "param", metric_col: "metric"}
            )
            sub = sub.sort_values("param")
            sub["ic"] = os.path.basename(ic_path)
            records.append(sub)
            used_param_col = chosen_param

        if not records:
            raise FileNotFoundError(
                f"No '{stats_filename}' files found in any IC folder under {base_dir}"
            )

        all_df = pd.concat(records, ignore_index=True)

        # Aggregate across ICs (mean and std per parameter value)
        grouped = all_df.groupby("param")["metric"]
        mean_series = grouped.mean()
        std_series = grouped.std(ddof=1).fillna(0.0)

        x = mean_series.index.values
        y = mean_series.values
        s = std_series.values

        y4_hi = y + 4 * s
        y6_hi = y + 6 * s

        # Get plot configuration
        plot_config = self.config.get('plot_options', {})
        fig_size = plot_config.get('figure_size', [8, 4.5])
        dpi = plot_config.get('dpi', 150)
        colors = plot_config.get('colors', {})
        
        # Set up colors with defaults
        color_individual = colors.get('individual_lines', '#6c757d')
        color_mean = colors.get('mean', '#1f77b4')
        color_4sigma = colors.get('envelope_4sigma', '#2ca02c')
        color_6sigma = colors.get('envelope_6sigma', '#ff7f0e')
        
        # Set up alphas with defaults
        alpha_individual = plot_config.get('alpha_individual', 0.35)
        alpha_mean = plot_config.get('alpha_mean', 0.10)
        alpha_4sigma = plot_config.get('alpha_envelope_4sigma', 0.20)
        alpha_6sigma = plot_config.get('alpha_envelope_6sigma', 0.10)

        plt.figure(figsize=fig_size)

        # Plot each realisation (IC) as dotted line with low opacity
        ic_groups = all_df.groupby("ic")
        first = True
        for ic_name, g in ic_groups:
            plt.plot(
                g["param"].values,
                g["metric"].values,
                linestyle=":",
                color=color_individual,
                alpha=alpha_individual,
                lw=1.0,
                label="realisations" if first else None,
            )
            first = False

        # One-sided upper envelopes with different colors
        plt.fill_between(x, y, y6_hi, color=color_6sigma, alpha=alpha_6sigma, label="+6σ")
        plt.fill_between(x, y, y4_hi, color=color_4sigma, alpha=alpha_4sigma, label="+4σ")

        # Mean line
        plt.plot(x, y, color=color_mean, ls="--", label="mean", alpha=alpha_mean)

        plt.xlabel(used_param_col if used_param_col is not None else "parameter")
        plt.ylabel(metric_col)
        plt.title(f"{model.upper()}: {metric_col} vs parameter (mean, +4σ, +6σ across ICs)")
        plt.grid(True, ls="--", alpha=0.3)
        plt.legend(loc="best", frameon=False)
        plt.tight_layout()

        if save:
            out_dir = os.path.join("plots", model)
            os.makedirs(out_dir, exist_ok=True)
            out_name = f"{metric_col}_envelope_across_ics.png"
            out_path = os.path.join(out_dir, out_name)
            plt.savefig(out_path, dpi=dpi)
            print(f"Saved: {out_path}")

        # Show plot based on config
        show_plots = plot_config.get('show_plots', True)
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_metrics(self, model: str):
        """Plot various metrics for the model."""
        if not self.config[model]['analysis'].get('plot_metrics', False):
            print(f"⏭️  Skipping metric plotting for {model.upper()}")
            return
        
        print(f"📊 Plotting {model.upper()} metrics...")
        
        # Get plot options from config
        plot_options = self.config.get('plot_options', {})
        enabled_metrics = plot_options.get('enabled_metrics', ['avg_degree', 'max_degree'])
        
        try:
            # Plot each enabled metric
            for metric in enabled_metrics:
                try:
                    # Try weighted first, then unweighted; prefer .bin, fallback to .csv
                    stats_files = [
                        "weighted_degree_stats.bin",
                        "unweighted_degree_stats.bin",
                        "weighted_degree_stats.csv",
                        "unweighted_degree_stats.csv",
                    ]
                    plotted = False
                    
                    for stats_file in stats_files:
                        try:
                            self.plot_metric_envelope_across_ics(
                                model, 
                                metric, 
                                stats_filename=stats_file,
                                save=plot_options.get('save_plots', True)
                            )
                            plotted = True
                            break
                        except (FileNotFoundError, ValueError) as e:
                            continue
                    
                    if not plotted:
                        print(f"⚠️  Warning: Could not find metric '{metric}' in any stats file for {model}")
                        
                except Exception as e:
                    print(f"⚠️  Warning: Failed to plot metric '{metric}' for {model}: {e}")
                    
            print(f"✓ {model.upper()} metric plotting completed")
        except Exception as e:
            print(f"⚠️  Warning: Failed to plot metrics for {model}: {e}")

    def _read_stb1(self, path: str) -> pd.DataFrame:
        """Read an STB1 binary summary table into a pandas DataFrame."""
        with open(path, 'rb') as f:
            if f.read(4) != b'STB1':
                raise ValueError('Invalid STB1 magic')
            ncols = struct.unpack('<I', f.read(4))[0]
            cols = []
            for _ in range(int(ncols)):
                klen = struct.unpack('<H', f.read(2))[0]
                cols.append(f.read(int(klen)).decode('ascii'))
            nrows = struct.unpack('<I', f.read(4))[0]
            total = int(nrows) * int(ncols)
            buf = f.read(total * 8)
            if len(buf) != total * 8:
                raise ValueError('Unexpected EOF while reading STB1 data matrix')
            arr = np.frombuffer(buf, dtype='<f8').reshape((int(nrows), int(ncols)))
        return pd.DataFrame(arr, columns=cols)
    
    def analyze_model(self, model: str):
        """Run complete analysis pipeline for a specific model."""
        if not self.config[model]['enabled']:
            print(f"⏭️  Skipping {model.upper()} - disabled in configuration")
            return
        
        print(f"\n🔬 Starting {model.upper()} Analysis")
        print("=" * 50)
        
        # Check execution mode
        exec_mode = self.config.get('execution_mode', {})
        load_existing = exec_mode.get('load_existing_graphs', False)
        run_until_graphs = exec_mode.get('run_until_graphs', False)
        skip_simulations = exec_mode.get('skip_simulations', False)
        
        # Generate initial conditions and parameters
        initial_conditions = self.generate_initial_conditions(model)
        parameters = self.generate_parameter_range(model)
        
        if load_existing:
            print(f"📂 Loading existing graphs for {model.upper()}...")
            # Skip simulation and data loading, go directly to graph analysis
            self.construct_visibility_graphs(model, initial_conditions)
            self.analyze_degree_distributions(model, initial_conditions)
            self.plot_metrics(model)
        else:
            # Standard pipeline or partial execution
            if not skip_simulations:
                # Run simulations
                csv_paths = self.run_simulations(model, initial_conditions, parameters)
                
                # Load dataframes
                if model == 'fhn':
                    col_name = self.config[model]['parameters']['column_u']
                elif model == 'linard':
                    col_name = self.config[model]['parameters']['column_x']
                
                dataframes = self.load_dataframes(model, csv_paths, col_name)
                
                # Analysis pipeline
                self.process_time_series(model, initial_conditions, dataframes, parameters)
                self.generate_bifurcation_diagrams(model, initial_conditions, dataframes)
            
            # Graph construction and analysis
            self.construct_visibility_graphs(model, initial_conditions)
            
            # Stop here if run_until_graphs is True
            if run_until_graphs:
                print(f"🛑 Stopping after graph generation for {model.upper()} (run_until_graphs=True)")
                # Zip/prune only if metrics were computed locally and we are stopping here
                if os.getenv('GV_SKIP_METRICS', '0') != '1':
                    try:
                        self._zip_and_prune_ic_data(model, initial_conditions)
                    except Exception as e:
                        print(f"⚠️  Warning: Failed to zip/prune artifacts for {model.upper()}: {e}")
                return
            
            # Continue with post-graph analysis
            self.analyze_degree_distributions(model, initial_conditions)
            self.plot_metrics(model)

        # Now that all local analysis and plotting are done, zip and prune if metrics were computed locally
        if os.getenv('GV_SKIP_METRICS', '0') != '1':
            try:
                self._zip_and_prune_ic_data(model, initial_conditions)
            except Exception as e:
                print(f"⚠️  Warning: Failed to zip/prune artifacts for {model.upper()}: {e}")
        
        print(f"✅ {model.upper()} analysis completed!\n")
    
    def run_analysis(self, models: List[str] = None):
        """Run analysis for specified models or all enabled models."""
        if models is None:
            models = ['fhn', 'linard']
        
        print("🚀 Starting Chaotic Systems Analysis")
        print("=" * 60)
        
        for model in models:
            if model not in self.config:
                print(f"⚠️  Warning: Unknown model '{model}' - skipping")
                continue
            
            try:
                self.analyze_model(model)
            except Exception as e:
                print(f"❌ Error analyzing {model.upper()}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print("🎉 Analysis pipeline completed!")

    def _zip_and_prune_ic_data(self, model: str, initial_conditions: Tuple[np.ndarray, np.ndarray]):
        """Zip per-IC timeseries, weighted_graphs, unweighted_graphs and delete the originals.

        - Creates three zip files per IC in data/{model}/{ic1}_{ic2}:
          timeseries_<ic>.zip, weighted_graphs_<ic>.zip, unweighted_graphs_<ic>.zip
        - Only zips .bin artifacts; leaves metrics and summary tables as-is.
        """
        ic1, ic2 = initial_conditions
        for ic1_val, ic2_val in zip(ic1, ic2):
            ic_dir = f"data/{model}/{ic1_val}_{ic2_val}"
            if not os.path.isdir(ic_dir):
                continue
            # Gather files by category
            timeseries_files = sorted(glob(os.path.join(ic_dir, 'output_*.bin')))
            weighted_files = sorted(glob(os.path.join(ic_dir, 'weighted_graph_*.bin')))
            unweighted_files = sorted(glob(os.path.join(ic_dir, 'unweighted_graph_*.bin')))

            categories = [
                (timeseries_files, f"timeseries_{ic1_val}_{ic2_val}.zip"),
                (weighted_files, f"weighted_graphs_{ic1_val}_{ic2_val}.zip"),
                (unweighted_files, f"unweighted_graphs_{ic1_val}_{ic2_val}.zip"),
            ]

            for files, zipname in categories:
                if not files:
                    continue
                zip_path = os.path.join(ic_dir, zipname)
                with zipfile.ZipFile(zip_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
                    for fp in files:
                        zf.write(fp, arcname=os.path.basename(fp))
                # Remove originals after successful zip
                for fp in files:
                    try:
                        os.remove(fp)
                    except OSError:
                        pass
                print(f"Zipped {len(files)} files to {zip_path} and deleted originals")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Analyze chaotic systems using visibility graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.json',
        help='Path to configuration file (default: config.json)'
    )
    
    parser.add_argument(
        '--model', '-m',
        choices=['fhn', 'linard', 'all'],
        default='all',
        help='Which model(s) to analyze (default: all)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--run-until-graphs',
        action='store_true',
        help='Stop execution after generating graphs'
    )
    
    parser.add_argument(
        '--load-existing-graphs',
        action='store_true',
        help='Skip simulations and load existing graphs for analysis'
    )
    
    parser.add_argument(
        '--skip-simulations',
        action='store_true',
        help='Skip simulation step (assumes data already exists)'
    )
    
    args = parser.parse_args()
    
    # Set up matplotlib for non-interactive use if needed
    if not args.verbose:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    
    try:
        # Initialize analyzer
        analyzer = SystemAnalyzer(args.config)
        
        # Override config with command line arguments if provided
        if args.run_until_graphs:
            analyzer.config['execution_mode']['run_until_graphs'] = True
        if args.load_existing_graphs:
            analyzer.config['execution_mode']['load_existing_graphs'] = True
        if args.skip_simulations:
            analyzer.config['execution_mode']['skip_simulations'] = True
        
        # Determine which models to run
        if args.model == 'all':
            models = ['fhn', 'linard']
        else:
            models = [args.model]
        
        # Run analysis
        analyzer.run_analysis(models)
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
