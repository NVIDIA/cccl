#!/bin/bash

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ci_dir/pyenv_helper.sh"

# Parse common arguments
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"
cuda_major_version=$(nvcc --version | grep release | awk '{print $6}' | tr -d ',' | cut -d '.' -f 1 | cut -d 'V' -f 2)

# Setup Python environment
setup_python_env "${py_version}"

# Fetch or build the cuda_cccl wheel (base dependency):
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  wheel_artifact_name=$("$ci_dir/util/workflow/get_wheel_artifact_name.sh")
  "$ci_dir/util/artifacts/download.sh" ${wheel_artifact_name} /home/coder/cccl/
else
  "$ci_dir/build_cuda_cccl_python.sh" -py-version "${py_version}"
fi

# Install cuda_cccl base wheel
CUDA_CCCL_WHEEL_PATH="$(ls /home/coder/cccl/wheelhouse/cuda_cccl-*.whl)"
python -m pip install "${CUDA_CCCL_WHEEL_PATH}[cu${cuda_major_version}]"

# Fetch or build the experimental wheel:
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  experimental_artifact_name="${wheel_artifact_name}_experimental"
  "$ci_dir/util/artifacts/download.sh" ${experimental_artifact_name} /home/coder/cccl/
else
  "$ci_dir/build_cuda_cccl_experimental_python_experimental.sh" -py-version "${py_version}"
fi

# Install cuda_cccl_experimental wheel
EXPERIMENTAL_WHEEL_PATH="$(ls /home/coder/cccl/wheelhouse_experimental/cuda_cccl_experimental-*.whl)"
python -m pip install "${EXPERIMENTAL_WHEEL_PATH}[test-cu${cuda_major_version}]"

# Run tests for STF module
cd "/home/coder/cccl/python/cuda_cccl_experimental/tests/"
python -m pytest -n auto -v stf/
