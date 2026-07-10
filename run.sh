#!/usr/bin/env bash
set -euo pipefail

# Configuration parameters
PROVIDER=${PROVIDER:-"AIX"}      # SMARTSIM, AIX, PHYDLL
DEVICE=${DEVICE:-"GPU"}          # GPU, CPU
API_MODE=${API_MODE:-"STATIC"}   # STATIC, ORDERED, KEYED, ORDERED_MULTI, KEYED_MULTI
STEPS=${STEPS:-1}
CLIENTS=${CLIENTS:-1}
COMPILE=${COMPILE:-0}
MODEL=${MODEL:-"giant"}
BATCH_SIZE=${BATCH_SIZE:-1}
SCOREP_MPP=${SCOREP_MPP:-}
PHYDLL_PY_SCOREP_WRAPPER=${PHYDLL_PY_SCOREP_WRAPPER:-0}

export PROVIDER
export STEPS
export CLIENTS
export API_MODE
export MODEL
export MERGE_STRATEGY
export BATCH_SIZE
export PHYDLL_PY_SCOREP_WRAPPER

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(realpath "${SCRIPT_DIR}/..")"
PYTHON_RUNTIME_ROOT="${BASE_DIR}/CPP-ML-Interface/extern/python"

if [[ -n "${MODULE_TEST_BUILD_DIR:-}" ]]; then
    BUILD_STATE_FILE="${MODULE_TEST_BUILD_DIR}/build_state.env"
else
    BUILD_STATE_FILE="${SCRIPT_DIR}/build/build_state.env"
fi

# If USE_SCOREP was not specified by the caller, inherit the mode of the selected
# build. This avoids running a Score-P-linked binary as non-Score-P.
if [[ -z "${USE_SCOREP+x}" ]]; then
    if [[ -f "${BUILD_STATE_FILE}" ]]; then
        USE_SCOREP="$(grep '^USE_SCOREP=' "${BUILD_STATE_FILE}" | tail -n 1 | cut -d= -f2- || true)"
        USE_SCOREP="${USE_SCOREP:-0}"
    else
        USE_SCOREP=0
    fi
fi
export USE_SCOREP

if [[ -z "${SCOREP_MPP}" && -f "${BUILD_STATE_FILE}" ]]; then
    SCOREP_MPP="$(grep '^SCOREP_MPP=' "${BUILD_STATE_FILE}" | tail -n 1 | cut -d= -f2- || true)"
fi
SCOREP_MPP="${SCOREP_MPP:-mpi}"
export SCOREP_MPP

if [[ -z "${MODULE_TEST_BUILD_DIR:-}" ]]; then
    if [[ "${USE_SCOREP}" == "1" ]]; then
        if [[ "${SCOREP_MPP}" == "mpi" ]]; then
            MODULE_TEST_BUILD_DIR="${SCRIPT_DIR}/build-scorep"
        else
            MODULE_TEST_BUILD_DIR="${SCRIPT_DIR}/build-scorep-${SCOREP_MPP}"
        fi
    else
        MODULE_TEST_BUILD_DIR="${SCRIPT_DIR}/build"
    fi
fi
SOLVER_BIN="${MODULE_TEST_BUILD_DIR}/module_test_solver"

if [[ "${USE_SCOREP}" == "1" ]]; then
    AIXELERATOR_INSTALL_PREFIX="${BASE_DIR}/CPP-ML-Interface/extern/AIxeleratorService/INSTALL-SCOREP"
    SCOREP_BIN_DIR="$(dirname "$(command -v scorep-config)")"
    export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="${SCOREP_WRAPPER_INSTRUMENTER_FLAGS:---nocompiler --user --mpp=${SCOREP_MPP} --io=none --memory=malloc --thread=none --nocuda}"
    export SCOREP_ENABLE_PROFILING=true
    export SCOREP_ENABLE_TRACING=false
else
    AIXELERATOR_INSTALL_PREFIX="${BASE_DIR}/CPP-ML-Interface/extern/AIxeleratorService/INSTALL"
fi

# Source environment
source "${BASE_DIR}/set_env_claix23_cuda12.4.sh"

# Perform model conversion if needed
if [[ "${DEVICE}" == "CPU" && "${MODEL}" != "multi_input" ]]; then
    MODEL_SRC="${BASE_DIR}/mini_app/train_models/model_a/${MODEL}_cuda.pt"
    MODEL_DST="${BASE_DIR}/mini_app/train_models/model_a/${MODEL}_cpu.pt"
    if [[ -f "${MODEL_SRC}" ]]; then
        echo "Converting ${MODEL_SRC} to CPU model ${MODEL_DST}..."
        # Use the smartsim_cpu python for conversion
        "${PYTHON_RUNTIME_ROOT}/smartsim_cpu/bin/python" "${SCRIPT_DIR}/convert_to_cpu.py" "${MODEL_SRC}" "${MODEL_DST}"
    fi
fi

# Set SmartRedis timeouts for large models (matching smoke test logic)
export SR_MODEL_TIMEOUT=900000
export SR_CMD_TIMEOUT=900000
export SR_SOCKET_TIMEOUT=900000

# Select Python environment and SmartSim device string
if [[ "${DEVICE}" == "GPU" ]]; then
    RUNTIME_DEVICE="smartsim_cuda-12"
    USE_GPU=1
else
    RUNTIME_DEVICE="smartsim_cpu"
    USE_GPU=0
fi
SMARTSIM_PYTHON="${SMARTSIM_PYTHON:-${PYTHON_RUNTIME_ROOT}/${RUNTIME_DEVICE}/bin/python}"
PY_ENV="${PYTHON_RUNTIME_ROOT}/${RUNTIME_DEVICE}"

# Select appropriate config file
if [[ -z "${CONFIG_FILE:-}" ]]; then
    if [[ "${PROVIDER}" == "SMARTSIM" ]]; then
        if [[ "${DEVICE}" == "GPU" ]]; then
            CONFIG_FILE="${SCRIPT_DIR}/config_smartsim_gpu.toml"
        else
            CONFIG_FILE="${SCRIPT_DIR}/config_smartsim_cpu.toml"
        fi
    elif [[ "${PROVIDER}" == "AIX" ]]; then
        if [[ "${API_MODE}" == "STATIC" ]]; then
            if [[ "${DEVICE}" == "GPU" ]]; then
                CONFIG_FILE="${SCRIPT_DIR}/config_aix_gpu.toml"
            else
                CONFIG_FILE="${SCRIPT_DIR}/config_aix_cpu.toml"
            fi
        else
            if [[ "${DEVICE}" == "GPU" ]]; then
                CONFIG_FILE="${SCRIPT_DIR}/config_aix_gpu_flex.toml"
            else
                CONFIG_FILE="${SCRIPT_DIR}/config_aix_cpu_flex.toml"
            fi
        fi
    elif [[ "${PROVIDER}" == "PHYDLL" ]]; then
        if [[ "${DEVICE}" == "GPU" ]]; then
            CONFIG_FILE="${SCRIPT_DIR}/config_phydll_gpu.toml"
        else
            CONFIG_FILE="${SCRIPT_DIR}/config_phydll_cpu.toml"
        fi
    else
        echo "Unsupported provider: ${PROVIDER}" >&2
        exit 1
    fi
fi

echo "--- Run Configuration ---"
echo "Provider: ${PROVIDER}"
echo "Device:   ${DEVICE}"
echo "Clients:  ${CLIENTS}"
echo "Steps:    ${STEPS}"
echo "Config:   $(basename "${CONFIG_FILE}")"
echo "Python:   ${SMARTSIM_PYTHON}"
echo "--------------------------"

if [[ "${COMPILE}" -eq 1 ]]; then
    EXTRA_CMAKE_ARGS=()
    if [[ "${PROVIDER}" == "PHYDLL" ]]; then
        EXTRA_CMAKE_ARGS+=("-DWITH_PHYDLL=ON")
    fi
    EXTRA_CMAKE_ARGS+=("-DTORCH_VERSION=2.4.0" "-DWITH_SCOREP=OFF")
    EXTRA_CMAKE_ARGS+=("-DAIXELERATOR_PREBUILT_INSTALL_PREFIX=${AIXELERATOR_INSTALL_PREFIX}")
    EXTRA_CMAKE_ARGS+=("-DAIXELERATOR_PREBUILT_LIB_DIR=${AIXELERATOR_INSTALL_PREFIX}/lib")
    if [[ "${USE_SCOREP:-}" == "1" ]]; then
        EXTRA_CMAKE_ARGS+=("-DWITH_SCOREP=ON" "-DCPPML_SCOREP_MPP=${SCOREP_MPP}" "-DMODULE_TEST_SCOREP_MPP=${SCOREP_MPP}" "-DAIXELERATOR_CMAKE_ARGS=-DWITH_TORCH=ON -DWITH_SCOREP=ON -DBUILD_TESTS=OFF")
    else
        EXTRA_CMAKE_ARGS+=("-DAIXELERATOR_CMAKE_ARGS=-DWITH_TORCH=ON -DBUILD_TESTS=OFF")
    fi
    cmake -U AIXELERATOR_PREBUILT_LIB -S "${SCRIPT_DIR}" -B "${MODULE_TEST_BUILD_DIR}" \
            -DSMARTSIM_PYTHON="${SMARTSIM_PYTHON}" \
            "${EXTRA_CMAKE_ARGS[@]}"
    cmake --build "${MODULE_TEST_BUILD_DIR}" --target module_test_solver -j
        cat > "${MODULE_TEST_BUILD_DIR}/build_state.env" <<EOF
# Generated by module_test/run.sh COMPILE=1. Used by run.sh when USE_SCOREP is unset.
USE_SCOREP=${USE_SCOREP:-0}
SCOREP_MPP=${SCOREP_MPP}
WITH_AIX=1
WITH_PHYDLL=1
WITH_SMARTSIM=1
SMARTSIM_PYTHON=${SMARTSIM_PYTHON}
MODULE_TEST_BUILD_DIR=${MODULE_TEST_BUILD_DIR}
AIXELERATOR_PREBUILT_INSTALL_PREFIX=${AIXELERATOR_INSTALL_PREFIX}
EOF
fi

# 1. SMARTSIM Provider
if [[ "${PROVIDER}" == "SMARTSIM" ]]; then
    # Staging runtime libs if they exist
    RUNTIME_EXTRA_LIB_DIR="${PY_ENV}/runtime_libs"
    if [[ -d "${RUNTIME_EXTRA_LIB_DIR}" ]]; then
            export LD_LIBRARY_PATH="${RUNTIME_EXTRA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
            echo "Using runtime extra libs from ${RUNTIME_EXTRA_LIB_DIR}"
    fi

    ENDPOINT_FILE="${SCRIPT_DIR}/.ssdb_endpoint"
    DONE_FILE="${SCRIPT_DIR}/.solver_done"
    SS_PORT=${SS_PORT:-6780}

    rm -f "${ENDPOINT_FILE}" "${DONE_FILE}"

    if [[ -f "${ENDPOINT_FILE}" ]]; then
        export SSDB="$(tr -d '\n' < "${ENDPOINT_FILE}")"
    fi

    # Use the generalized controller from CPP-ML-Interface
    SMARTSIM_CONTROLLER="${SCRIPT_DIR}/../CPP-ML-Interface/dl_clients/smartsim_controller.py"
    
    CONTROLLER_ARGS=(
            "--endpoint-file" "${ENDPOINT_FILE}"
            "--done-file" "${DONE_FILE}"
            "--port" "${SS_PORT}"
            "--exp-dir" "${SCRIPT_DIR}/module_tests"
    )

    if [[ -n "${OMP_NUM_THREADS:-}" ]]; then
            CONTROLLER_ARGS+=("--intra-op-threads" "${OMP_NUM_THREADS}" "--cpu-cores-per-node" "${OMP_NUM_THREADS}")
    fi

    "${SMARTSIM_PYTHON}" "${SMARTSIM_CONTROLLER}" "${CONTROLLER_ARGS[@]}" &
    DRIVER_PID=$!

    cleanup() {
            if [[ -n "${DRIVER_PID:-}" ]] && kill -0 "${DRIVER_PID}" 2>/dev/null; then
                    touch "${DONE_FILE}" || true
                    wait "${DRIVER_PID}" || true
            fi
    }
    trap cleanup EXIT

    echo "Waiting for SmartSim database to start..."
    for _ in {1..120}; do
            if [[ -s "${ENDPOINT_FILE}" ]]; then
                    break
            fi
            sleep 0.5
    done

    if [[ ! -s "${ENDPOINT_FILE}" ]]; then
            echo "Timed out waiting for SmartSim endpoint file: ${ENDPOINT_FILE}" >&2
            exit 1
    fi

    export SSDB
    SSDB="$(tr -d '\n' < "${ENDPOINT_FILE}")"
    echo "Using SSDB=${SSDB}"

    mpirun -x MODULE_TEST_RUN_ID -x PROVIDER -x API_MODE -x STEPS -x MODEL -x MERGE_STRATEGY -x TIMING_LOG -x MLCOUPLING_INTRA_OP_THREADS -x MLCOUPLING_INTER_OP_THREADS -x BATCH_SIZE -n "${CLIENTS}" "${SOLVER_BIN}" "${CONFIG_FILE}"

    touch "${DONE_FILE}"
    wait "${DRIVER_PID}"

# 2. AIX Provider
elif [[ "${PROVIDER}" == "AIX" ]]; then
    # Force visibility for GPU if requested, but respect CUDA_VISIBLE_DEVICES if already set
    if [[ "${DEVICE}" == "GPU" ]]; then
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
    fi
    mpirun -x MODULE_TEST_RUN_ID -x PROVIDER -x API_MODE -x STEPS -x MODEL -x MERGE_STRATEGY -x TIMING_LOG -x MLCOUPLING_INTRA_OP_THREADS -x MLCOUPLING_INTER_OP_THREADS -x BATCH_SIZE -n "${CLIENTS}" "${SOLVER_BIN}" "${CONFIG_FILE}"

# 3. PHYDLL Provider
elif [[ "${PROVIDER}" == "PHYDLL" ]]; then
    USE_PYTHON_DL_CLIENT=${USE_PYTHON_DL_CLIENT:-0}
    PHYDLL_PY_SCOREP_WRAPPER=${PHYDLL_PY_SCOREP_WRAPPER:-0}
    DL_CLIENT_CMD=()
    if [[ "${USE_PYTHON_DL_CLIENT}" == "1" ]]; then
            if [[ "${USE_SCOREP}" == "1" && "${PHYDLL_PY_SCOREP_WRAPPER}" == "1" ]]; then
                SCOREP_BIN_DIR="$(dirname "$(command -v scorep-config)")"
                DL_CLIENT_CMD=("env" "PATH=${SCOREP_BIN_DIR}:${PATH}" "${SMARTSIM_PYTHON}" "-m" "scorep" "--keep-files" "--instrumenter-type=dummy" "--noinstrumenter" "--mpp=${SCOREP_MPP}" "${SCRIPT_DIR}/../CPP-ML-Interface/dl_clients/phydll_dl_client.py")
            else
                DL_CLIENT_CMD=("${SMARTSIM_PYTHON}" "${SCRIPT_DIR}/../CPP-ML-Interface/dl_clients/phydll_dl_client.py")
            fi
            PHYDLL_REBUILD_DL_CLIENT=0
    else
            if [[ "${USE_SCOREP}" == "1" ]]; then
                if [[ "${SCOREP_MPP}" == "mpi" ]]; then
                    PHYDLL_DL_BUILD_DIR_DEFAULT="${SCRIPT_DIR}/../CPP-ML-Interface/dl_clients/build-scorep"
                else
                    PHYDLL_DL_BUILD_DIR_DEFAULT="${SCRIPT_DIR}/../CPP-ML-Interface/dl_clients/build-scorep-${SCOREP_MPP}"
                fi
            else
                PHYDLL_DL_BUILD_DIR_DEFAULT="${SCRIPT_DIR}/../CPP-ML-Interface/dl_clients/build-module-test"
            fi
            PHYDLL_DL_BUILD_DIR="${PHYDLL_DL_BUILD_DIR:-${PHYDLL_DL_BUILD_DIR_DEFAULT}}"
            PHYDLL_DL_CLIENT="${PHYDLL_DL_CLIENT:-${PHYDLL_DL_BUILD_DIR}/phydll_dl_client}"
            DL_CLIENT_CMD=("${PHYDLL_DL_CLIENT}")
            PHYDLL_REBUILD_DL_CLIENT=${PHYDLL_REBUILD_DL_CLIENT:-1}
    fi
    
    NP_PHY=${CLIENTS}
    NP_DL=1
    PHYDLL_DL_COUNT=1
    export PHYDLL_DL_COUNT

    # Rebuild DL client if requested
    if [[ "${USE_PYTHON_DL_CLIENT}" == "0" ]]; then
            if [[ "${PHYDLL_REBUILD_DL_CLIENT}" == "1" || ! -x "${PHYDLL_DL_CLIENT}" ]]; then
                    if [[ "${USE_SCOREP}" == "1" ]]; then
                        cmake -S "${SCRIPT_DIR}/../CPP-ML-Interface/dl_clients" -B "${PHYDLL_DL_BUILD_DIR}" -DWITH_SCOREP=ON -DCPPML_SCOREP_MPP="${SCOREP_MPP}"
                    else
                        cmake -S "${SCRIPT_DIR}/../CPP-ML-Interface/dl_clients" -B "${PHYDLL_DL_BUILD_DIR}" -DWITH_SCOREP=OFF
                    fi
                    cmake --build "${PHYDLL_DL_BUILD_DIR}" -j
            fi
            if [[ ! -x "${PHYDLL_DL_CLIENT}" ]]; then
                    echo "PHYDLL_DL_CLIENT not executable: ${PHYDLL_DL_CLIENT}" >&2
                    exit 1
            fi
    fi

    MPIRUN_ENV=(-x MODULE_TEST_RUN_ID -x PROVIDER -x API_MODE -x STEPS -x MODEL -x MERGE_STRATEGY -x TIMING_LOG -x MLCOUPLING_INTRA_OP_THREADS -x MLCOUPLING_INTER_OP_THREADS -x BATCH_SIZE)
    MPIRUN_ENV+=(-x USE_SCOREP -x SCOREP_MPP)
    MPIRUN_ENV+=(-x PHYDLL_PY_SCOREP_WRAPPER)
    if [[ "${USE_SCOREP}" == "1" ]]; then
        MPIRUN_ENV+=(-x SCOREP_ENABLE_PROFILING -x SCOREP_ENABLE_TRACING)
    fi
    if [[ -n "${SCOREP_METRIC_PAPI_SEP:-}" ]]; then
        MPIRUN_ENV+=(-x SCOREP_METRIC_PAPI_SEP)
    fi
    if [[ -n "${SCOREP_METRIC_PAPI:-}" ]]; then
        MPIRUN_ENV+=(-x SCOREP_METRIC_PAPI)
    fi
    if [[ "${DEVICE}" == "GPU" ]]; then
        MPIRUN_ENV+=(-x CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0})
        MPIRUN_ENV+=(-x CUDA_DEVICE_ORDER)
    fi
    MPIRUN_ENV+=(-x PHYDLL_DL_COUNT)

    PHY_APP_ENV=("${MPIRUN_ENV[@]}")
    DL_APP_ENV=("${MPIRUN_ENV[@]}")

    # Add libphydll.so to LD_LIBRARY_PATH for the DL client.
    # Use the Score-P build tree when running the Score-P PHYDLL configuration.
    if [[ "${USE_SCOREP}" == "1" ]]; then
        PHYDLL_LIB_BUILD_DIR_DEFAULT="${SCRIPT_DIR}/../CPP-ML-Interface/extern/phydll/build-SCOREP"
    else
        PHYDLL_LIB_BUILD_DIR_DEFAULT="${SCRIPT_DIR}/../CPP-ML-Interface/extern/phydll/build"
    fi
    PHYDLL_LIB_BUILD_DIR="${PHYDLL_LIB_BUILD_DIR:-${PHYDLL_LIB_BUILD_DIR_DEFAULT}}"
    PHYDLL_LIB_DIR=$(realpath "${PHYDLL_LIB_BUILD_DIR}/lib")
    PHY_APP_ENV+=(-x LD_LIBRARY_PATH="${PHYDLL_LIB_DIR}:${LD_LIBRARY_PATH:-}")
    DL_APP_ENV+=(-x LD_LIBRARY_PATH="${PHYDLL_LIB_DIR}:${LD_LIBRARY_PATH:-}")

    # MPMD split: solver uses color=0; DL client uses MPI_UNDEFINED (no MPI_APPNUM reliance).
    echo "Launching PhyDLL with NP_PHY=${NP_PHY}, NP_DL=${NP_DL}, using ${DEVICE}"
    mpirun --bind-to none "${PHY_APP_ENV[@]}" -n "${NP_PHY}" "${SOLVER_BIN}" "${CONFIG_FILE}" : "${DL_APP_ENV[@]}" -n "${NP_DL}" "${DL_CLIENT_CMD[@]}"
fi
