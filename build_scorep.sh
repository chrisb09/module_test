#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="${SCOREP_WRAPPER_INSTRUMENTER_FLAGS:---nocompiler --user --mpp=none --io=none --memory=malloc --thread=none --nocuda}"
USE_SCOREP=1 "${SCRIPT_DIR}/build.sh" "$@"
