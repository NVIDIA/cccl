#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/build_common.sh"

print_environment_details

PRESET="thrust"

CMAKE_OPTIONS="-DCMAKE_CXX_STANDARD=$CXX_STANDARD -DCMAKE_CUDA_STANDARD=$CXX_STANDARD"

configure_and_build_preset "Thrust" "$PRESET" "$CMAKE_OPTIONS"

# Create test artifacts:
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    run_command "📦  Packaging test artifacts" /home/coder/cccl/ci/upload_thrust_test_artifacts.sh
fi

print_time_summary
