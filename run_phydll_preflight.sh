#!/bin/bash
#SBATCH --job-name=phydll_preflight
#SBATCH --time=00:50:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=phydll_preflight_%j.log
#SBATCH --error=phydll_preflight_%j.err

# Focused preflight for the packed/uniform_chunks matrix axis:
#   PHYDLL, CPU, no Score-P, one workload/batch, both DL clients, both layouts,
#   one normal model + the MMCP transformer (exercises dl_count=2 in chunk mode).

set -euo pipefail

cd /hpcwork/ro092286/smartsim/module_test
source ../set_env_claix23_cuda12.4.sh

python3 test_matrix.py \
    --providers PHYDLL \
    --devices CPU \
    --models perfect mmcp_transformer \
    --api-modes STATIC KEYED_MULTI \
    --workloads 1/1 \
    --batch-sizes 1 \
    --scorep off \
    --transport-layouts packed uniform_chunks \
    --out "${SLURM_JOB_ID}_preflight.md"
