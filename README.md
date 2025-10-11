# Graph-Visibility

FHN system, Linard System.

1e5 time series. 
201 parameter values. 

constructing weighted and unweighted visibility graphs from all parameter values
calculating metrics for unweighted graph and averaging across realisations. 
calculating metrics for coarsened weighted graph and averaging across realisations. 

# Script and Project Structure
# Chaotic Systems Visibility Graph Analysis

This project provides a comprehensive analysis pipeline for chaotic dynamical systems using visibility graphs. It has been converted from a Jupyter notebook to a modular, configurable Python script for better reproducibility and automation.

## Features

- **JSON Configuration**: All parameters configurable via `config.json`
- **Multiple Models**: Support for FHN (FitzHugh-Nagumo) and Liénard oscillators
- **Modular Pipeline**: Each analysis step can be enabled/disabled independently
- **Comprehensive Analysis**:
  - Time series simulation and processing
  - Bifurcation diagram generation
  - Visibility graph construction (weighted and unweighted)
  - Degree distribution analysis
  - Statistical envelope plotting across initial conditions
- **Command Line Interface**: Easy to run and integrate into automated workflows

## Quick Start

### 1. Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Compile C++ executables
make all

# Create necessary directories
make dirs
```

### 2. Basic Usage

```bash
# Run complete analysis for both models
python analyze_systems.py

# Run only FHN model
python analyze_systems.py --model fhn

# Run only Liénard model  
python analyze_systems.py --model linard

# Use custom configuration
python analyze_systems.py --config my_config.json

# Enable verbose output
python analyze_systems.py --verbose
```

### 3. Alternative Quick Runner

```bash
# Simple wrapper script
python run_analysis.py --model fhn
```

## Configuration

The analysis is controlled by `config.json`. Key sections:

### General Settings
```json
{
  "general": {
    "random_seed": 42,           // For reproducibility
    "stabilizing_time": 40000,   // Transient time to skip
    "window_size": 1,            // Local optima detection window
    "precision_degree": 6,       // Decimal precision
    "num_params": 101            // Number of parameter points
  },
  "simulation": {
    "fhn_num_steps": 200000,     // Number of steps for FHN simulations
    "linard_num_steps": 10000    // Number of steps for Linard simulations
  }
}
```

### Model Configuration
```json
{
  "fhn": {
    "enabled": true,             // Enable/disable this model
    "initial_conditions": {      // Random IC generation
      "u_count": 5,
      "v_count": 5,
      "u_base": -0.2,
      "u_range": 0.001,
      "v_base": -0.002,
      "v_range": 0.0001
    },
    "parameters": {              // Parameter sweep settings
      "A_min": 0.62,
      "A_max": 0.63,
      "column_u": "u",           // Column names in CSV
      "column_v": "v"
    },
    "analysis": {                // Enable/disable analysis steps
      "process_timeseries": true,
      "bifurcation_diagram": true,
      "construct_graphs": true,
      "weighted_degree_summary": true,
      "unweighted_degree_summary": true,
      "plot_metrics": true
    }
  }
}
```

## Project Structure

```
├── analyze_systems.py      # Main analysis script
├── run_analysis.py         # Simple wrapper runner
├── config.json            # Configuration file
├── requirements.txt       # Python dependencies
├── Makefile              # C++ compilation
├── cpputils/             # C++ source files
│   ├── fhn.cpp
│   ├── linard.cpp
│   ├── weighted_construct.cpp
│   └── unweighted_construct.cpp
├── pyutils/              # Python utility modules
│   ├── appendlocal.py
│   ├── processts.py
│   ├── bifurcation.py
│   └── graphmetrics.py
├── data/                 # Generated simulation data
├── plots/                # Generated plots and figures
└── unit_tests/           # Test suite
```

## Analysis Pipeline

The script follows this pipeline for each enabled model:

1. **Setup**: Compile executables, create directories, set random seeds
2. **Generate Initial Conditions**: Random IC generation based on config
3. **Generate Parameters**: Parameter range generation
4. **Run Simulations**: Execute C++ simulations for all IC/parameter combinations
5. **Load Data**: Load CSV files and append local optima information
6. **Process Time Series**: Generate time series plots and analyses
7. **Bifurcation Diagrams**: Create bifurcation diagrams
8. **Construct Graphs**: Build visibility graphs (weighted/unweighted)
9. **Analyze Degrees**: Compute degree statistics
10. **Plot Metrics**: Generate metric envelope plots across ICs

## Output Files

### Data Files
- `data/{model}/{ic1}_{ic2}/output_{param}.bin` - Raw simulation data (TSB1)
- `data/{model}/{ic1}_{ic2}/*degree_stats.bin` - Degree statistics tables (STB1)

### Plots
- `plots/{model}/{ic1}_{ic2}/` - Time series and analysis plots per IC
- `plots/{model}/bifurcation.png` - Bifurcation diagrams
- `plots/{model}/*_envelope_across_ics.png` - Metric envelopes

### Graph Files
- `data/{model}/{ic1}_{ic2}/weighted_graph_*.bin` (WGB1)
- `data/{model}/{ic1}_{ic2}/unweighted_graph_*.bin` (UGB1)
- Per-graph metrics: `*_metrics.bin` (MET1)

## Binary formats and migration

All pipeline artifacts now use compact binary files. Readers remain backward-compatible with legacy CSV when present, but writers emit .bin by default.

- TSB1 (Time Series Binary)
  - Layout: `magic='TSB1' (4 bytes)`, `uint32 cols` (=3), `uint32 rows`, followed by `rows` records of 3 x `float64` in order: `[time, u/x, v/y]`.
- UGB1 (Unweighted Graph Binary)
  - Layout: `magic='UGB1'`, `uint64 edge_count`, then for each edge: `int32 u`, `int32 v` (undirected; each edge stored once).
- WGB1 (Weighted Graph Binary)
  - Layout: `magic='WGB1'`, `uint64 edge_count`, then for each edge: `int32 u`, `int32 v`, `float64 weight`.
- MET1 (Metrics Binary)
  - Layout: `magic='MET1'`, `uint32 item_count`, then for each item: `uint16 key_len`, `key bytes (ASCII)`, `float64 value`.
- STB1 (Stats Table Binary)
  - Layout: `magic='STB1'`, `uint32 column_count`, then for each column: `uint16 name_len`, `name bytes (ASCII)`, `uint32 row_count`, followed by row-major `float64` values of shape `(row_count, column_count)`.

All integer counts use little-endian unsigned 32-bit except `edge_count` in graphs, which uses 64-bit to support large graphs. Floating point values are little-endian float64.

Migration notes:
- Existing CSV inputs will still be read where applicable, but new runs will generate `.bin` files.
- Update any ad hoc scripts to look for `.bin` extensions (see `run_hpc.sh` for examples).

## Customization

### Simulation Parameters

The number of simulation steps can be configured independently for each model:

```json
{
  "simulation": {
    "fhn_num_steps": 200000,     // Longer simulations for FHN
    "linard_num_steps": 10000    // Shorter simulations for Linard
  }
}
```

**C++ Command Line Interface:**
- FHN: `./fhn <A_value> <u0_value> <v0_value> [num_steps]`
- Linard: `./linard <omega_value> <x0_value> <y0_value> [num_steps]`

If `num_steps` is not provided, the executables use default values (FHN: 200,000, Linard: 10,000).

### Adding New Models

1. Add model configuration to `config.json`
2. Create corresponding C++ executable in `cpputils/`
3. Add model-specific logic in `analyze_systems.py` if needed
4. Include simulation step configuration in the `simulation` section

### Modifying Analysis Steps

Edit the `analysis` section in `config.json` to enable/disable specific steps:

```json
"analysis": {
  "process_timeseries": false,    // Skip time series processing
  "bifurcation_diagram": true,    // Generate bifurcation diagrams
  "construct_graphs": true,       // Build visibility graphs
  "plot_metrics": false          // Skip metric plotting
}
```

### Parameter Customization

Modify parameter ranges, initial condition ranges, and other settings directly in `config.json`.

## Performance Notes

- **Parallelization**: The C++ simulations run sequentially but could be parallelized
- **Memory Usage**: Large parameter sweeps may require significant memory
- **Disk Space**: Each simulation generates CSV files; monitor disk usage
- **Computation Time**: Full analysis can take significant time for large parameter ranges

## Troubleshooting

### Common Issues

1. **Compilation Errors**:
   ```bash
   make clean
   make all
   ```

2. **Missing Python Modules**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Permission Issues**:
   ```bash
   chmod +x analyze_systems.py run_analysis.py
   ```

4. **Configuration Errors**:
   - Check JSON syntax in `config.json`
   - Validate parameter ranges and paths

### Debug Mode

Run with verbose flag to see detailed output:
```bash
python analyze_systems.py --verbose
```

