#!/bin/bash
#SBATCH --job-name=phydll_repro
#SBATCH --time=00:20:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --output=phydll_repro_%j.log
#SBATCH --error=phydll_repro_%j.err

# Reproduce the Python+Score-P+2-client mmcp timeout on a compute node.
# Runs the same failing case with PHYDLL_PY_SCOREP_WRAPPER=1 vs =0, each under
# a 200s timeout, to isolate whether the Python scorep wrapper causes the hang.

set -u
cd /hpcwork/ro092286/smartsim/module_test
export CMI_DIR=/rwthfs/rz/cluster/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference/CPP-ML-Interface
export USE_SCOREP=1
export SCOREP_MPP=none
export SMARTSIM_PYTHON="${CMI_DIR}/extern/python/smartsim_cuda-12/bin/python"
source ../set_env_claix23_cuda12.4.sh

export SCOREP_EXPERIMENT_DIRECTORY=/hpcwork/ro092286/smartsim/module_test/scorep/repro

COMMON="PROVIDER=PHYDLL DEVICE=CPU API_MODE=ORDERED_MULTI STEPS=5 CLIENTS=2 BATCH_SIZE=1 MODEL=mmcp_transformer USE_SCOREP=1 SCOREP_MPP=none USE_PYTHON_DL_CLIENT=1 PHYDLL_REBUILD_DL_CLIENT=0 PHYDLL_DL_FIELD_COUNT=1 PHYDLL_DL_COUNT=1 COMPILE=0"

echo "===== WRAPPER=1 (failing matrix config) ====="
timeout 200 env ${COMMON} PHYDLL_PY_SCOREP_WRAPPER=1 ./run.sh > /tmp/repro_wrapper1.log 2>&1
echo "wrapper=1 exit=$?"

echo "===== WRAPPER=0 (solver Score-P on, no python wrapper) ====="
timeout 200 env ${COMMON} PHYDLL_PY_SCOREP_WRAPPER=0 ./run.sh > /tmp/repro_wrapper0.log 2>&1
echo "wrapper=0 exit=$?"

echo "===== wrapper1 log tail ====="
tail -25 /tmp/repro_wrapper1.log
echo "===== wrapper0 log tail ====="
tail -25 /tmp/repro_wrapper0.log
