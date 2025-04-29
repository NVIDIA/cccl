#!/bin/bash

set -euo pipefail

source "$(dirname "$0")/build_common.sh"

print_environment_details

fail_if_no_gpu

source "test_python_common.sh"

list_environment

# Install the wheel from the artifact location
WHEEL_PATH="../../artifacts/wheelhouse/cuda_cooperative-*.whl"
run_tests_from_wheel "cuda_cooperative" "$WHEEL_PATH"
