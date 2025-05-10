#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/pyenv_helper.sh"

cd "$(dirname "$0")/../python/cuda_cccl"

# Get the Python version from the command line arguments e.g., -py-version=3.10
py_version=${2#*=}
echo "Python version: ${py_version}"

# Setup Python environment
setup_python_env "${py_version}"

# Build the wheel and output to the wheelhouse directory
python -m pip wheel --no-deps . && \
cp *.whl /wheelhouse/
