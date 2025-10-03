#!/bin/bash
#SBATCH --job-name=graph-visibility
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=compute

# HPC Batch Script for Graph Visibility Analysis
# Usage: sbatch run_hpc.sh [config_file] [model] [additional_args]

# Default values
CONFIG_FILE="${1:-config.json}"
MODEL="${2:-all}"
ADDITIONAL_ARGS="${@:3}"

# Create necessary directories
mkdir -p logs data plots

echo "=========================================="
echo "Graph Visibility Analysis - HPC Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Config: $CONFIG_FILE"
echo "Model: $MODEL"
echo "Started at: $(date)"
echo "=========================================="

# Load required modules (adjust based on your HPC environment)
# module load gcc/11.2.0
# module load python/3.9.0

# Set up UV environment
if ! command -v uv &> /dev/null; then
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Create and activate virtual environment with UV
echo "Setting up Python environment with UV..."
uv venv --python 3.9 .venv
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv pip install -e .

# Compile C++ executables
echo "Compiling C++ executables..."
make clean
make all

# Create necessary directories
echo "Creating directories..."
make dirs

# Set environment variables for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the analysis
echo "Starting analysis..."
echo "Command: python analyze_systems.py --config $CONFIG_FILE --model $MODEL $ADDITIONAL_ARGS"

python analyze_systems.py --config "$CONFIG_FILE" --model "$MODEL" $ADDITIONAL_ARGS

EXIT_CODE=$?

echo "=========================================="
echo "Analysis completed with exit code: $EXIT_CODE"
echo "Ended at: $(date)"
echo "=========================================="

# Optional: Archive results
if [ $EXIT_CODE -eq 0 ]; then
    echo "Archiving results..."
    tar -czf "results_${SLURM_JOB_ID}.tar.gz" data plots logs
    echo "Results archived to results_${SLURM_JOB_ID}.tar.gz"
fi

exit $EXIT_CODE
