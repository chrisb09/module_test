#!/usr/bin/env bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=01:00:00
#SBATCH --job-name=build-module-test
#SBATCH --output=build-module-test.%j.out
#SBATCH --error=build-module-test.%j.err

# Exit on any error
set -euo pipefail

# Get directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_ML_DIR="$(realpath "${SCRIPT_DIR}/../CPP-ML-Interface")"
ABS_SCRIPT="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

# If not already running inside a Slurm job, self-submit via srun
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Not inside a Slurm job. Re-executing via srun on devel partition with 96 cores..."
    exec srun --partition=devel --cpus-per-task=96 --time=01:00:00 "${ABS_SCRIPT}" "$@"
fi

echo "=== Slurm Build Job Started ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "CPUs allocated: ${SLURM_CPUS_ON_NODE:-96}"

# 1. cd into CPP-ML-Interface folder
cd "${CPP_ML_DIR}"

# 2. source install.sh cuda-12
echo "Sourcing environment and runtime from CPP-ML-Interface/install.sh cuda-12..."
source ./install.sh cuda-12

# 3. go back to the original dir and run the build.sh
cd "${SCRIPT_DIR}"
echo "Running build.sh in ${SCRIPT_DIR}..."
./build.sh "$@"

echo "=== Slurm Build Job Completed Successfully ==="
