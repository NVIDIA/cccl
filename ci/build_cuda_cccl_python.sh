#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../python/cuda_cccl"

# Get the Python version from the command line arguments -py-version=3.10
py_version=${2#*=}
echo "Python version: ${py_version}"

# Build the wheel and output to the wheelhouse directory
ls $HOST_WORKSPACE/
python -m pip wheel --no-deps . && \
mv *.whl $HOST_WORKSPACE/wheelhouse/
