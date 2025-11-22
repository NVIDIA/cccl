#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/build_common.sh"

print_environment_details

PRESET="thrust"

CMAKE_OPTIONS="-DCMAKE_CXX_STANDARD=$CXX_STANDARD -DCMAKE_CUDA_STANDARD=$CXX_STANDARD"

GPU_REQUIRED=false

configure_and_build_preset "Thrust" "$PRESET" "$CMAKE_OPTIONS"
# Fail tests run compilers to check for build errors. Run those here as they can be
# very slow on GPU runners.
test_preset "Thrust" "thrust-fail" ${GPU_REQUIRED}

# Create test artifacts:
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    run_command "📦  Packaging test artifacts" /home/coder/cccl/ci/upload_thrust_test_artifacts.sh
fi

print_time_summary
