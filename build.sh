#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(realpath "${SCRIPT_DIR}/..")"

# Source environment
source "${BASE_DIR}/set_env_claix23_cuda12.4.sh"

# Use the smartsim_cuda-12 environment by default for building
SMARTSIM_PYTHON="${BASE_DIR}/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python"

cmake -S "${SCRIPT_DIR}" -B "${SCRIPT_DIR}/build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DWITH_AIX=ON \
      -DWITH_PHYDLL=ON \
      -DWITH_SMARTSIM=ON \
      -DSMARTSIM_PYTHON="${SMARTSIM_PYTHON}" \
      -DTORCH_CUDA_ARCH_LIST="9.0"

cmake --build "${SCRIPT_DIR}/build" -j 4
