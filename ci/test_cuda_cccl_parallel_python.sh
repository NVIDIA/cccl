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

# Fetch or build the cuda_cccl wheel:
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  wheel_artifact_name=$("$ci_dir/util/workflow/get_wheel_artifact_name.sh")
  "$ci_dir/util/artifacts/download.sh" ${wheel_artifact_name} /home/coder/cccl/
else
  "$ci_dir/build_cuda_cccl_python.sh" -py-version "${py_version}"
fi

# Install cuda_cccl
CUDA_CCCL_WHEEL_PATH="$(ls /home/coder/cccl/wheelhouse/cuda_cccl-*.whl)"
python -m pip install "${CUDA_CCCL_WHEEL_PATH}[test-cu${cuda_major_version}]"

# Run tests for parallel module
cd "/home/coder/cccl/python/cuda_cccl/tests/"
python -m pytest -n 6 -v parallel/ -m "not large"
python -m pytest -n 0 -v parallel/ -m "large"
