#!/usr/bin/env bash

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ci_dir/pyenv_helper.sh"

source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"
cuda_major_version=$(nvcc --version | grep release | awk '{print $6}' | tr -d ',' | cut -d '.' -f 1 | cut -d 'V' -f 2)
if [[ -z "${cuda_major_version}" ]]; then
  echo "Failed to detect CUDA major version from nvcc" >&2
  exit 1
fi

setup_python_env "${py_version}"

# This lane installs exactly one cuda_cccl wheel for the selected Python/CUDA
# environment. Missing artifacts can happen on unsupported platforms; multiple
# matches usually mean wheelhouse contains stale wheels from different builds.
find_cuda_cccl_wheel() {
  local wheelhouse="/home/coder/cccl/wheelhouse"
  local wheels=("${wheelhouse}"/cuda_cccl-*.whl)

  if [[ ! -e "${wheels[0]}" ]]; then
    echo "No cuda_cccl wheel found in ${wheelhouse}" >&2
    exit 1
  fi

  if [[ "${#wheels[@]}" -ne 1 ]]; then
    echo "Expected exactly one cuda_cccl wheel in ${wheelhouse}, found ${#wheels[@]}:" >&2
    printf '  %s\n' "${wheels[@]}" >&2
    exit 1
  fi

  echo "${wheels[0]}"
}

# Fetch or build the cuda_cccl wheel:
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  wheel_artifact_name=$("$ci_dir/util/workflow/get_wheel_artifact_name.sh")
  "$ci_dir/util/artifacts/download.sh" "${wheel_artifact_name}" /home/coder/cccl/
else
  "$ci_dir/build_cuda_cccl_python.sh" -py-version "${py_version}"
fi

# Install cuda_cccl with the test-cuXX extra
CUDA_CCCL_WHEEL_PATH="$(find_cuda_cccl_wheel)"
python -m pip install "${CUDA_CCCL_WHEEL_PATH}[test-cu${cuda_major_version}]"

# Run STF tests
cd "/home/coder/cccl/python/cuda_cccl/tests/"
python -m pytest -n auto -v stf/
