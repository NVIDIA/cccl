#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/pyenv_helper.sh"

# Get the Python version from the command line arguments -py-version=3.10
py_version=${2#*=}
echo "Python version: ${py_version}"

# Setup Python environment
setup_python_env "${py_version}"

cd /home/coder/cccl/python/cuda_cooperative

# Build the cccl wheel and copy it to the wheelhouse directory
python -m pip wheel --no-deps /home/coder/cccl/python/cuda_cccl && \
cp cuda_cccl-*.whl /home/coder/cccl/wheelhouse

# Build the wheel and output to the wheelhouse directory
python -m pip wheel --no-deps . && \
cp cuda_cooperative-*.whl /home/coder/cccl/wheelhouse
