#!/bin/bash

set -euo pipefail

source "$(dirname "$0")/build_common.sh"

print_environment_details

fail_if_no_gpu

source "test_python_common.sh"

list_environment

# Install the wheel from the artifact location
WHEEL_PATH="../../wheelhouse/cuda_parallel-*.whl"
run_tests_from_wheel "cuda_parallel" "$WHEEL_PATH"
