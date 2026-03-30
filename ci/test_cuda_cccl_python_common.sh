#!/bin/bash

set -euo pipefail

function install_cuda_cccl_wheel_for_tests() {
  local wheel_path="$1"
  local cuda_major_version="$2"

  local numba_cuda_git_url="${NUMBA_CUDA_GIT_URL:-https://github.com/NVIDIA/numba-cuda.git}"
  local numba_cuda_git_ref="${NUMBA_CUDA_GIT_REF:-}"
  local numba_cuda_extra="cu${cuda_major_version}"

  if [[ -n "${numba_cuda_git_ref}" ]]; then
    echo "Installing numba-cuda override from ${numba_cuda_git_url}@${numba_cuda_git_ref}"
    python -m pip install       "numba-cuda[${numba_cuda_extra}] @ git+${numba_cuda_git_url}@${numba_cuda_git_ref}"

    python -m pip install       pytest       pytest-xdist       pytest-benchmark

    if [[ "${cuda_major_version}" == "12" ]]; then
      python -m pip install cupy-cuda12x
    elif [[ "${cuda_major_version}" == "13" ]]; then
      python -m pip install cupy-cuda13x
    fi

    python -m pip install --no-deps "${wheel_path}"
    return
  fi

  python -m pip install "${wheel_path}[test-cu${cuda_major_version}]"
}
