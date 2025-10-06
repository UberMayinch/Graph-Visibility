#!/bin/bash
#SBATCH --job-name=graph-visibility
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2048
#SBATCH --time=72:00:00
#SBATCH --mail-user=chinmay.sharma@research.iiit.ac.in
#SBATCH --mail-type=ALL

# HPC Batch Script for Optimized Graph Visibility Analysis
# Usage: sbatch run_hpc.sh [config_file] [model] [additional_args]
# 
# This script runs the fully optimized analysis pipeline with:
# - Parallel C++ compilation with aggressive optimization flags
# - Multi-threaded simulation execution  
# - Optimized graph construction and metrics calculation
# - Efficient memory and I/O management

set -e  # Exit on any error

# Default values
CONFIG_FILE="${1:-config.json}"
MODEL="${2:-all}"
ADDITIONAL_ARGS="${@:3}"

# Create necessary directories
mkdir -p logs data plots

echo "=========================================="
echo "Optimized Graph Visibility Analysis - HPC Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Config: $CONFIG_FILE"
echo "Model: $MODEL"
echo "Started at: $(date)"
echo "=========================================="

# Validate config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "❌ Error: Configuration file '$CONFIG_FILE' not found!"
    echo "Available config files:"
    ls -la *.json 2>/dev/null || echo "No .json files found"
    exit 1
fi

# Load required modules (adjust based on your HPC environment)
module load u22/cmake/4.0.1 
module load u22/python/3.12.4 

# Set up UV environment (no modules loaded)
echo "Setting up UV package manager..."
if ! command -v uv &> /dev/null; then
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
    if ! command -v uv &> /dev/null; then
        echo "❌ Error: UV installation failed!"
        exit 1
    fi
fi

# Create and activate virtual environment with UV
echo "Creating Python virtual environment..."
uv venv --python 3.12 .venv
if [[ ! -f ".venv/bin/activate" ]]; then
    echo "❌ Error: Virtual environment creation failed!"
    exit 1
fi

source .venv/bin/activate
echo "✓ Activated virtual environment"

# Install dependencies with no cache to save space
echo "Installing Python dependencies..."
uv pip install --no-cache-dir -r requirements.txt
if [[ $? -ne 0 ]]; then
    echo "❌ Error: Dependency installation failed!"
    exit 1
fi
echo "✓ Dependencies installed successfully"

# Compile C++ executables with optimizations
echo "Compiling optimized C++ executables..."
make clean
make all
if [[ $? -ne 0 ]]; then
    echo "❌ Error: C++ compilation failed!"
    exit 1
fi

# Verify all executables exist
REQUIRED_EXES=("fhn" "linard" "weighted_construct" "unweighted_construct" "graph_metrics")
for exe in "${REQUIRED_EXES[@]}"; do
    if [[ ! -x "$exe" ]]; then
        echo "❌ Error: Executable '$exe' not found or not executable!"
        exit 1
    fi
done
echo "✓ All executables compiled successfully"

# Create necessary directories
echo "Creating analysis directories..."
make dirs
echo "✓ Directories created successfully"

# Set environment variables for optimal performance
echo "Setting performance environment variables..."
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK

echo "✓ Using $OMP_NUM_THREADS threads for parallel processing"


# Run the optimized analysis
echo "=========================================="
echo "Starting optimized analysis pipeline..."
echo "Command: python analyze_systems.py --config $CONFIG_FILE --model $MODEL $ADDITIONAL_ARGS --verbose"
echo "=========================================="

START_TIME=$(date +%s)
python analyze_systems.py --config "$CONFIG_FILE" --model "$MODEL" $ADDITIONAL_ARGS --verbose
EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "=========================================="
echo "Analysis completed!"
echo "Exit code: $EXIT_CODE"
echo "Duration: ${DURATION} seconds ($(($DURATION / 60))m $(($DURATION % 60))s)"
echo "Ended at: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Analysis completed successfully!"
    
    # Show result summary
    echo ""
    echo "📊 Results Summary:"
    echo "Data files: $(find data -name '*.csv' | wc -l)"
    echo "Graph files: $(find data -name '*graph*.csv' | wc -l)"
    echo "Metrics files: $(find data -name '*metrics*.csv' | wc -l)"
    echo "Plot files: $(find plots -name '*.png' 2>/dev/null | wc -l)"
    
    # Archive results
    echo ""
    echo "📦 Archiving results..."
    tar -czf "results_${SLURM_JOB_ID}.tar.gz" data plots logs
    ARCHIVE_SIZE=$(du -h "results_${SLURM_JOB_ID}.tar.gz" | cut -f1)
    echo "✓ Results archived to results_${SLURM_JOB_ID}.tar.gz (${ARCHIVE_SIZE})"
else
    echo "❌ Analysis failed with exit code: $EXIT_CODE"
    echo "Check the error logs above for details."
fi

echo "=========================================="
exit $EXIT_CODE
