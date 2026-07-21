#!/usr/bin/env bash

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ci_dir/pyenv_helper.sh"

# Parse common arguments
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"
# Extract CTK major.minor version from nvcc
cuda_version=$(nvcc --version | grep release | awk '{print $6}' | tr -d ',' | cut -d '.' -f 1-2 | cut -d 'V' -f 2)
cuda_major_version=$(echo "$cuda_version" | cut -d '.' -f 1)
# Pin cuda-toolkit wheels to the container's CTK minor. A lane can set
# CCCL_PYTHON_TEST_LATEST_CTK=1 to skip the pin and instead test whatever pip
# resolves as the latest minor -- what a plain `pip install` (no lockfile) gets.
if [[ "${CCCL_PYTHON_TEST_LATEST_CTK:-}" != "1" ]]; then
  export PIP_CONSTRAINT="${TMPDIR:-/tmp}/ctk-constraint.txt"
  echo "cuda-toolkit==${cuda_version}.*" > "$PIP_CONSTRAINT"
fi

# Setup Python environment
setup_python_env "${py_version}"

# Fetch or build the cuda_cccl wheel:
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  wheel_artifact_name=$("$ci_dir/util/workflow/get_wheel_artifact_name.sh")
  "$ci_dir/util/artifacts/download.sh" "${wheel_artifact_name}" /home/coder/cccl/
else
  "$ci_dir/build_cuda_cccl_python.sh" -py-version "${py_version}"
fi

# Install cuda_cccl
CUDA_CCCL_WHEEL_PATH="$(ls /home/coder/cccl/wheelhouse/cuda_cccl-*.whl)"
python -m pip install "${CUDA_CCCL_WHEEL_PATH}[test-cu${cuda_major_version}]"

# Run tests for coop module
cd "/home/coder/cccl/python/cuda_cccl/tests/"
python -m pytest -n auto -v coop/_experimental/
