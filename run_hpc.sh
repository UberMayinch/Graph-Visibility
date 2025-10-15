#!/bin/bash
#SBATCH --job-name=graph-visibility
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err
#SBATCH --mem-per-cpu=2048
#SBATCH --time=72:00:00
#SBATCH --mincpus=32
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

# Default values (capture before changing directories)
CONFIG_FILE_INPUT="${1:-config.json}"
MODEL="${2:-all}"
ADDITIONAL_ARGS="${@:3}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

# Prepare scratch workspace and clone repo there
SCRATCH_BASE="/scratch/chinmay.sharma"
JOB_SCRATCH="$SCRATCH_BASE/Graph-Visibility-$SLURM_JOB_ID"
echo "Setting up scratch workspace at: $JOB_SCRATCH"
mkdir -p "$SCRATCH_BASE"
cd "$SCRATCH_BASE"

REPO_URL_SSH="git@github.com:UberMayinch/Graph-Visibility.git"
REPO_URL_HTTPS="https://github.com/UberMayinch/Graph-Visibility.git"

echo "Cloning repository to scratch (SSH)..."
if git clone --depth 1 "$REPO_URL_SSH" "$JOB_SCRATCH" 2>/dev/null; then
    echo "✓ Cloned via SSH"
else
    echo "SSH clone failed; trying HTTPS..."
    if git clone --depth 1 "$REPO_URL_HTTPS" "$JOB_SCRATCH" 2>/dev/null; then
        echo "✓ Cloned via HTTPS"
    else
        echo "Git clone failed; falling back to copying current working tree to scratch"
        rsync -a --exclude '.git' "$SUBMIT_DIR/" "$JOB_SCRATCH/"
    fi
fi

cd "$JOB_SCRATCH"

# Ensure logs dir exists in scratch for tar
mkdir -p logs

# Resolve config path relative to submit dir and copy into scratch repo
if [[ -f "$SUBMIT_DIR/$CONFIG_FILE_INPUT" ]]; then
    CONFIG_BASENAME="$(basename "$CONFIG_FILE_INPUT")"
    cp -f "$SUBMIT_DIR/$CONFIG_FILE_INPUT" "$JOB_SCRATCH/$CONFIG_BASENAME"
    CONFIG_FILE="$CONFIG_BASENAME"
else
    CONFIG_FILE="config.json"
fi

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

# Validate config file exists in scratch
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

# Create necessary directories in scratch
echo "Creating analysis directories..."
make dirs
echo "✓ Directories created successfully"

# Set environment variables for optimal performance
echo "Setting performance environment variables..."
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK

echo "✓ Configured metrics to run sequentially (OMP_NUM_THREADS=$OMP_NUM_THREADS)"


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
    echo "Data files: $(find -L data -name '*.bin' | wc -l)"
    echo "Graph files: $(find -L data -name '*graph*.bin' | wc -l)"
    echo "Metrics files: $(find -L data -name '*metrics*.bin' | wc -l)"
    echo "Plot files: $(find -L plots -name '*.png' 2>/dev/null | wc -l)"
    
    # Archive results from scratch
    echo ""
    echo "📦 Archiving results..."
    tar -chzf "results_${SLURM_JOB_ID}.tar.gz" data plots logs
    ARCHIVE_SIZE=$(du -h "results_${SLURM_JOB_ID}.tar.gz" | cut -f1)
    echo "✓ Results archived to results_${SLURM_JOB_ID}.tar.gz (${ARCHIVE_SIZE})"

    # # Copy archive back to submission directory
    # cp -f "results_${SLURM_JOB_ID}.tar.gz" "$SUBMIT_DIR/" || true
    # echo "✓ Copied archive to $SUBMIT_DIR/results_${SLURM_JOB_ID}.tar.gz"

    # Parallel metrics fan-out: run multiple sequential metrics jobs concurrently
    echo ""
    echo "🧮 Parallel metrics fan-out stage..."
    JOBS=${METRICS_JOBS:-$SLURM_CPUS_PER_TASK}
    if [[ -z "$JOBS" || "$JOBS" -lt 1 ]]; then JOBS=1; fi
    MISSING=$(find -L data -type f -name '*graph*.bin' ! -name '*_metrics.bin' | wc -l)
    echo "Missing metrics: $MISSING | Concurrency: $JOBS"
    if [[ "$MISSING" -gt 0 ]]; then
        COUNT=0
        while IFS= read -r GRAPH; do
            METRICS="${GRAPH%.bin}_metrics.bin"
            if [[ -f "$METRICS" ]]; then
                continue
            fi
            ./graph_metrics "$GRAPH" "$METRICS" &
            COUNT=$((COUNT+1))
            if (( COUNT % JOBS == 0 )); then
                wait
            fi
        done < <(find -L data -type f -name '*graph*.bin' ! -name '*_metrics.bin' | sort)
        wait
        echo "✓ Parallel metrics fan-out completed."
    else
        echo "No missing metrics to process."
    fi
else
    echo "❌ Analysis failed with exit code: $EXIT_CODE"
    echo "Check the error logs above for details."
fi

echo "=========================================="
exit $EXIT_CODE
