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

# Locate exactly one wheel matching the given glob under the shared wheelhouse.
find_one_wheel() {
  local glob="$1"
  local wheelhouse="/home/coder/cccl/wheelhouse"
  local wheels=("${wheelhouse}"/${glob})

  if [[ ! -e "${wheels[0]}" ]]; then
    echo "No wheel matching '${glob}' found in ${wheelhouse}" >&2
    exit 1
  fi

  if [[ "${#wheels[@]}" -ne 1 ]]; then
    echo "Expected exactly one wheel matching '${glob}' in ${wheelhouse}, found ${#wheels[@]}:" >&2
    printf '  %s\n' "${wheels[@]}" >&2
    exit 1
  fi

  echo "${wheels[0]}"
}

# Fetch or build the cuda_cccl and cuda_stf wheels. cuda-stf depends on
# cuda-cccl for cuda.cccl.headers and cuda.cccl._cuda_version_utils, and the
# interop tests exercise cuda.compute, so both wheels are required.
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  cccl_artifact_name=$("$ci_dir/util/workflow/get_wheel_artifact_name.sh")
  "$ci_dir/util/artifacts/download.sh" "${cccl_artifact_name}" /home/coder/cccl/
  stf_artifact_name=$(CCCL_WHEEL_KIND=stf "$ci_dir/util/workflow/get_wheel_artifact_name.sh")
  "$ci_dir/util/artifacts/download.sh" "${stf_artifact_name}" /home/coder/cccl/
else
  "$ci_dir/build_cuda_cccl_python.sh" -py-version "${py_version}"
  "$ci_dir/build_cuda_stf_python.sh" -py-version "${py_version}"
fi

# Install cuda_cccl first (provides cuda.cccl.headers, cuda.compute), then
# cuda_stf with its test extra. cuda-stf's unpinned cuda-cccl dependency is
# satisfied by the already-installed local wheel.
CUDA_CCCL_WHEEL_PATH="$(find_one_wheel 'cuda_cccl-*.whl')"
python -m pip install "${CUDA_CCCL_WHEEL_PATH}[test-cu${cuda_major_version}]"

CUDA_STF_WHEEL_PATH="$(find_one_wheel 'cuda_stf-*.whl')"
python -m pip install "${CUDA_STF_WHEEL_PATH}[test-cu${cuda_major_version}]"

# Run STF tests and examples
cd "/home/coder/cccl/python/cuda_stf/tests/"
python -m pytest -n auto -v stf/
python -m pytest -n 6 -v test_examples.py
