#!/bin/bash

set -euo pipefail
source "$(dirname "$0")/pyenv_helper.sh"

# Get the Python version from the command line arguments -py-version=3.10
py_version=${2#*=}
echo "Python version: ${py_version}"

# Setup Python environment
setup_python_env "${py_version}"

# Install cuda_cccl
CUDA_CCCL_WHEEL_PATH="$(ls /home/coder/cccl/wheelhouse/cuda_cccl-*.whl)"
python -m pip install "${CUDA_CCCL_WHEEL_PATH}[test]"

# Run tests
cd "/home/coder/cccl/python/cuda_cccl/tests/"
python -m pytest -n auto -v
