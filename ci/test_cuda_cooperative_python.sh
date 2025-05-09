#!/bin/bash

set -euo pipefail

# Get the Python version from the command line arguments -py-version=3.10
py_version=${2#*=}
echo "Python version: ${py_version}"

# Source common scripts and check environment
source "$(dirname "$0")/build_common.sh"
print_environment_details
fail_if_no_gpu

source "$(dirname "$0")/test_python_common.sh"
list_environment

# Install the wheel and run tests
CUDA_COOPERATIVE_WHEEL_PATH="$(ls /workspace/wheelhouse/cuda_cooperative-*.whl)"
python -m pip install "${CUDA_COOPERATIVE_WHEEL_PATH}[test]"
cd "$(dirname "$0")/../python/cuda_cooperative/tests/"
python -m pytest -n ${PARALLEL_LEVEL} -v
