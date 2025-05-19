#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/pyenv_helper.sh"

# Get the Python version from the command line arguments e.g., -py-version=3.10
py_version=${2#*=}
echo "Python version: ${py_version}"

# Setup Python environment
setup_python_env "${py_version}"

# Build the wheel and output to the wheelhouse directory
cd /home/coder/cccl/python/cuda_cccl
python -m pip wheel --no-deps . && cp *.whl /home/coder/cccl/wheelhouse
