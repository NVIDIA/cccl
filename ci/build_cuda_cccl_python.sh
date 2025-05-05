#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../python/cuda_cccl"

# Get the Python version from the command line arguments -py-version=3.10
py_version=${1#*=}
echo "Python version: ${py_version}"
