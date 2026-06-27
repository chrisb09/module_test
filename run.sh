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

export PROVIDER
export STEPS
export CLIENTS
export API_MODE
export MODEL
export MERGE_STRATEGY
export BATCH_SIZE

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(realpath "${SCRIPT_DIR}/..")"
PYTHON_RUNTIME_ROOT="${BASE_DIR}/CPP-ML-Interface/extern/python"

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

# Compile if requested
if [[ "${COMPILE}" -eq 1 ]]; then
    cmake -S "${SCRIPT_DIR}" -B "${SCRIPT_DIR}/build" \
            -DSMARTSIM_PYTHON="${SMARTSIM_PYTHON}"
    cmake --build "${SCRIPT_DIR}/build" -j
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

    mpirun -x MODULE_TEST_RUN_ID -x PROVIDER -x API_MODE -x STEPS -x MODEL -x MERGE_STRATEGY -x TIMING_LOG -x MLCOUPLING_INTRA_OP_THREADS -x MLCOUPLING_INTER_OP_THREADS -x BATCH_SIZE -n "${CLIENTS}" "${SCRIPT_DIR}/build/module_test_solver" "${CONFIG_FILE}"

    touch "${DONE_FILE}"
    wait "${DRIVER_PID}"

# 2. AIX Provider
elif [[ "${PROVIDER}" == "AIX" ]]; then
    # Force visibility for GPU if requested, but respect CUDA_VISIBLE_DEVICES if already set
    if [[ "${DEVICE}" == "GPU" ]]; then
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
    fi
    mpirun -x MODULE_TEST_RUN_ID -x PROVIDER -x API_MODE -x STEPS -x MODEL -x MERGE_STRATEGY -x TIMING_LOG -x MLCOUPLING_INTRA_OP_THREADS -x MLCOUPLING_INTER_OP_THREADS -x BATCH_SIZE -n "${CLIENTS}" "${SCRIPT_DIR}/build/module_test_solver" "${CONFIG_FILE}"

# 3. PHYDLL Provider
elif [[ "${PROVIDER}" == "PHYDLL" ]]; then
    USE_PYTHON_DL_CLIENT=${USE_PYTHON_DL_CLIENT:-0}
    DL_CLIENT_CMD=()
    if [[ "${USE_PYTHON_DL_CLIENT}" == "1" ]]; then
            DL_CLIENT_CMD=("${SMARTSIM_PYTHON}" "${SCRIPT_DIR}/../CPP-ML-Interface/dl_clients/phydll_dl_client.py")
            PHYDLL_REBUILD_DL_CLIENT=0
    else
            PHYDLL_DL_CLIENT="${PHYDLL_DL_CLIENT:-${SCRIPT_DIR}/../CPP-ML-Interface/dl_clients/build/phydll_dl_client}"
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
                    cmake -S "${SCRIPT_DIR}/../CPP-ML-Interface/dl_clients" -B "${SCRIPT_DIR}/../CPP-ML-Interface/dl_clients/build"
                    cmake --build "${SCRIPT_DIR}/../CPP-ML-Interface/dl_clients/build" -j
            fi
            if [[ ! -x "${PHYDLL_DL_CLIENT}" ]]; then
                    echo "PHYDLL_DL_CLIENT not executable: ${PHYDLL_DL_CLIENT}" >&2
                    exit 1
            fi
    fi

    MPIRUN_ENV=(-x MODULE_TEST_RUN_ID -x PROVIDER -x API_MODE -x STEPS -x MODEL -x MERGE_STRATEGY -x TIMING_LOG -x MLCOUPLING_INTRA_OP_THREADS -x MLCOUPLING_INTER_OP_THREADS -x BATCH_SIZE)
    if [[ "${DEVICE}" == "GPU" ]]; then
        MPIRUN_ENV+=(-x CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0})
        MPIRUN_ENV+=(-x CUDA_DEVICE_ORDER)
    fi
    MPIRUN_ENV+=(-x PHYDLL_DL_COUNT)

    PHY_APP_ENV=("${MPIRUN_ENV[@]}")
    DL_APP_ENV=("${MPIRUN_ENV[@]}")

    # Add libphydll.so to LD_LIBRARY_PATH for the DL client
    PHYDLL_LIB_DIR=$(realpath "${SCRIPT_DIR}/../CPP-ML-Interface/extern/phydll/build/lib")
    DL_APP_ENV+=(-x LD_LIBRARY_PATH="${PHYDLL_LIB_DIR}:${LD_LIBRARY_PATH:-}")

    # MPMD split: solver uses color=0; DL client uses MPI_UNDEFINED (no MPI_APPNUM reliance).
    echo "Launching PhyDLL with NP_PHY=${NP_PHY}, NP_DL=${NP_DL}, using ${DEVICE}"
    mpirun --bind-to none "${PHY_APP_ENV[@]}" -n "${NP_PHY}" "${SCRIPT_DIR}/build/module_test_solver" "${CONFIG_FILE}" : "${DL_APP_ENV[@]}" -n "${NP_DL}" "${DL_CLIENT_CMD[@]}"
fi
