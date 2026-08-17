#!/bin/bash
#SBATCH --job-name=phydll_repro_sp
#SBATCH --time=00:15:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --output=phydll_repro_sp_%j.log
#SBATCH --error=phydll_repro_sp_%j.err

# Faithful reproduction of the Score-P python mmcp 2-client failures on a
# compute node (the Score-P bash -c wrapper is mangled by login-node OpenMPI).
# Uses the rebuilt scorep solver + smartsim CMI python + GCC14 runtime libs.

set -u
cd /hpcwork/ro092286/smartsim/module_test
export USE_SCOREP=1
export SCOREP_MPP=none
source ../set_env_claix23_cuda12.4.sh >/dev/null 2>&1
export GCC14_ROOT=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/GCCcore/14.2.0
export LD_LIBRARY_PATH="${GCC14_ROOT}/lib64:${LD_LIBRARY_PATH}"
export SMARTSIM_PYTHON=/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python

export SCOREP_EXPERIMENT_DIRECTORY=/hpcwork/ro092286/smartsim/module_test/scorep/repro_sp

echo "===== packed mmcp python 5/2/1 scorep=ON ====="
timeout 240 env PROVIDER=PHYDLL DEVICE=CPU API_MODE=ORDERED_MULTI STEPS=5 CLIENTS=2 BATCH_SIZE=1 \
    MODEL=mmcp_transformer USE_SCOREP=1 SCOREP_MPP=none USE_PYTHON_DL_CLIENT=1 \
    PHYDLL_PY_SCOREP_WRAPPER=1 PHYDLL_REBUILD_DL_CLIENT=0 \
    PHYDLL_DL_FIELD_COUNT=1 PHYDLL_DL_COUNT=1 COMPILE=0 ./run.sh > /tmp/repro_sp_packed.log 2>&1
echo "packed exit=$?"
tail -20 /tmp/repro_sp_packed.log
