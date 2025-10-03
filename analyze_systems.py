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

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random

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
    
    def run_simulations(self, model: str, initial_conditions: Tuple[np.ndarray, np.ndarray], 
                       parameters: np.ndarray) -> Dict[Tuple[float, float], Dict[float, str]]:
        """Run simulations for all parameter values and initial conditions."""
        ic1, ic2 = initial_conditions
        csv_paths = defaultdict(dict)
        
        print(f"🚀 Running {model.upper()} simulations...")
        
        # Create paths dictionary
        for ic1_val, ic2_val in zip(ic1, ic2):
            for param in parameters:
                csv_paths[(ic1_val, ic2_val)][param] = f"data/{model}/{ic1_val}_{ic2_val}/output_{param}.csv"
        
        # Run simulations
        for ic1_val, ic2_val in zip(ic1, ic2):
            # Create directory for this initial condition
            make_dir_cmd = f"mkdir -p -- data/{model}/{ic1_val}_{ic2_val}"
            subprocess.run(make_dir_cmd, shell=True)
            
            # Run simulation for each parameter value
            for param in parameters:
                # Get number of steps from configuration
                num_steps = self.config['simulation'][f'{model}_num_steps']
                cmd = f"./{model} {param} {ic1_val} {ic2_val} {num_steps}"
                try:
                    subprocess.run(cmd, shell=True, check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    print(f"⚠️  Warning: Simulation failed for {model} with params {param}, {ic1_val}, {ic2_val}")
                    continue
        
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
    
    def construct_visibility_graphs(self, model: str, initial_conditions: Tuple[np.ndarray, np.ndarray]):
        """Construct visibility graphs."""
        if not self.config[model]['analysis']['construct_graphs']:
            print(f"⏭️  Skipping graph construction for {model.upper()}")
            return
        
        print(f"🕸️  Constructing {model.upper()} visibility graphs...")
        
        ic1, ic2 = initial_conditions
        constructGraphs(ic1, ic2, model)
        
        print(f"✓ {model.upper()} visibility graphs completed")
    
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
                                       stats_filename: str = "unweighted_degree_stats.csv",
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
            csv_path = os.path.join(ic_path, stats_filename)
            if not os.path.isfile(csv_path):
                continue

            df = pd.read_csv(csv_path)
            # Infer parameter column
            chosen_param = None
            for cand in candidate_param_cols:
                if cand in df.columns:
                    chosen_param = cand
                    break
            if chosen_param is None:
                numeric_cols = [c for c in df.columns if c != metric_col and pd.api.types.is_numeric_dtype(df[c])]
                if not numeric_cols:
                    raise ValueError(f"Could not infer parameter column in {csv_path}")
                chosen_param = numeric_cols[0]

            if metric_col not in df.columns:
                raise ValueError(f"Metric column '{metric_col}' not found in {csv_path}. Available: {list(df.columns)}")

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

        plt.figure(figsize=(8, 4.5))

        # Plot each realisation (IC) as dotted line with low opacity
        ic_groups = all_df.groupby("ic")
        first = True
        for ic_name, g in ic_groups:
            plt.plot(
                g["param"].values,
                g["metric"].values,
                linestyle=":",
                color="#6c757d",
                alpha=0.35,
                lw=1.0,
                label="realisations" if first else None,
            )
            first = False

        # One-sided upper envelopes with different colors
        plt.fill_between(x, y, y6_hi, color="#ff7f0e", alpha=0.10, label="+6σ")
        plt.fill_between(x, y, y4_hi, color="#2ca02c", alpha=0.20, label="+4σ")

        # Mean line
        plt.plot(x, y, color="#1f77b4", ls="--", label="mean", alpha=0.10)

        plt.xlabel(used_param_col if used_param_col is not None else "parameter")
        plt.ylabel(metric_col)
        plt.title(f"{model}: {metric_col} vs parameter (mean, +4σ, +6σ across ICs)")
        plt.grid(True, ls="--", alpha=0.3)
        plt.legend(loc="best", frameon=False)
        plt.tight_layout()

        if save:
            out_dir = os.path.join("plots", model)
            os.makedirs(out_dir, exist_ok=True)
            out_name = f"{metric_col}_envelope_across_ics.png"
            out_path = os.path.join(out_dir, out_name)
            plt.savefig(out_path, dpi=150)
            print(f"Saved: {out_path}")

        plt.show()
    
    def plot_metrics(self, model: str):
        """Plot various metrics for the model."""
        if not self.config[model]['analysis'].get('plot_metrics', False):
            print(f"⏭️  Skipping metric plotting for {model.upper()}")
            return
        
        print(f"📊 Plotting {model.upper()} metrics...")
        
        try:
            # Plot average and max degree envelopes
            self.plot_metric_envelope_across_ics(model, "avg_degree")
            self.plot_metric_envelope_across_ics(model, "max_degree")
            print(f"✓ {model.upper()} metric plotting completed")
        except Exception as e:
            print(f"⚠️  Warning: Failed to plot metrics for {model}: {e}")
    
    def analyze_model(self, model: str):
        """Run complete analysis pipeline for a specific model."""
        if not self.config[model]['enabled']:
            print(f"⏭️  Skipping {model.upper()} - disabled in configuration")
            return
        
        print(f"\n🔬 Starting {model.upper()} Analysis")
        print("=" * 50)
        
        # Generate initial conditions and parameters
        initial_conditions = self.generate_initial_conditions(model)
        parameters = self.generate_parameter_range(model)
        
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
        self.construct_visibility_graphs(model, initial_conditions)
        self.analyze_degree_distributions(model, initial_conditions)
        self.plot_metrics(model)
        
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
    
    args = parser.parse_args()
    
    # Set up matplotlib for non-interactive use if needed
    if not args.verbose:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    
    try:
        # Initialize analyzer
        analyzer = SystemAnalyzer(args.config)
        
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
