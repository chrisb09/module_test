#!/bin/bash
#SBATCH --job-name=phydll_rerun_fail
#SBATCH --time=00:45:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --output=rerun_failures_%j.log
#SBATCH --error=rerun_failures_%j.err

# Re-run exactly the previously failing matrix block: PHYDLL python scorep=on
# mmcp_transformer 5/2 with ORDERED_MULTI/KEYED_MULTI on CPU/GPU, both layouts.
# (Runs the cpp variants too as a regression check; they were passing.)

set -u
cd /hpcwork/ro092286/smartsim/module_test
export USE_SCOREP=1
export SCOREP_MPP=none
bash -c 'source ../set_env_claix23_cuda12.4.sh >/dev/null 2>&1
export GCC14_ROOT=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/GCCcore/14.2.0
export LD_LIBRARY_PATH="${GCC14_ROOT}/lib64:${LD_LIBRARY_PATH}"
export SMARTSIM_PYTHON=/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python
python3 test_matrix.py \
    --providers PHYDLL \
    --devices CPU GPU \
    --models mmcp_transformer \
    --api-modes ORDERED_MULTI KEYED_MULTI \
    --workloads 5/2 \
    --batch-sizes 1 7 \
    --scorep on \
    --transport-layouts packed uniform_chunks \
    --out "${SLURM_JOB_ID}_rerun_failures.md"'
