#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/pyenv_helper.sh"

cd "$(dirname "$0")/../python/cuda_cooperative"

# Get the Python version from the command line arguments -py-version=3.10
py_version=${2#*=}
echo "Python version: ${py_version}"

# Setup Python environment
setup_python_env "${py_version}"

# Build the cccl wheel and copy it to the wheelhouse directory
python -m pip wheel --no-deps ../cuda_cccl && \
cp cuda_cccl-*.whl /wheelhouse/

# Build the wheel and output to the wheelhouse directory
python -m pip wheel --no-deps . && \
cp cuda_cooperative-*.whl /wheelhouse/
