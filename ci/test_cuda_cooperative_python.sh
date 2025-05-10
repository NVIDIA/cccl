#!/bin/bash

set -euo pipefail
source "$(dirname "$0")/pyenv_helper.sh"

# Get the Python version from the command line arguments -py-version=3.10
py_version=${2#*=}
echo "Python version: ${py_version}"

# Setup Python environment
setup_python_env "${py_version}"

# Install the wheel and run tests
CUDA_CCCL_WHEEL_PATH="$(ls /wheelhouse/cuda_cccl-*.whl)"
python -m pip install "${CUDA_CCCL_WHEEL_PATH}"
CUDA_COOPERATIVE_WHEEL_PATH="$(ls /wheelhouse/cuda_cooperative-*.whl)"
python -m pip install "${CUDA_COOPERATIVE_WHEEL_PATH}[test]"
cd "$(dirname "$0")/../python/cuda_cooperative/tests/"
python -m pytest -n auto -v
