#!/usr/bin/env bash

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
  "$ci_dir/util/artifacts/download.sh" "${wheel_artifact_name}" /home/coder/cccl/
else
  "$ci_dir/build_cuda_cccl_python.sh" -py-version "${py_version}"
fi

# Install cuda_cccl
CUDA_CCCL_WHEEL_PATH="$(ls /home/coder/cccl/wheelhouse/cuda_cccl-*.whl)"
python -m pip install "${CUDA_CCCL_WHEEL_PATH}[test-cu${cuda_major_version}]"

# Run tests for compute module.
# On the v2 (HostJIT) backend, abort on first failure — the suite is still
# stabilizing and a single early failure is enough signal to investigate
# without scrolling through hundreds of subsequent passes.
pytest_extra=()
if [[ "${CCCL_PYTHON_USE_V2:-}" =~ ^(1|true|TRUE|on|ON)$ ]]; then
  pytest_extra+=(-x)
fi

cd "/home/coder/cccl/python/cuda_cccl/tests/"
if [[ "${CCCL_PYTHON_USE_V2:-}" =~ ^(1|true|TRUE|on|ON)$ ]]; then
  # The test isolates itself in a fresh subprocess (LLVM initialization is
  # process-wide and only cold once), but it carries the free_threading marker,
  # so it must be selected by node-id here or the sweeps below never run it.
  python -m pytest "${pytest_extra[@]}" -n 0 -v \
    compute/test_free_threading_stress.py::test_v2_concurrent_cold_llvm_initialization
fi
python -m pytest "${pytest_extra[@]}" -n 6 -v compute/ -m "not large and not free_threading"
python -m pytest "${pytest_extra[@]}" -n 0 -v compute/ -m "large and not free_threading"
