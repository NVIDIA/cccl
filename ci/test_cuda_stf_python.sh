#!/usr/bin/env bash

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ci_dir/pyenv_helper.sh"

source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found on PATH; cannot determine the CUDA version for cuda-stf extras" >&2
  exit 1
fi
# 'nvcc --version' prints e.g. "Cuda compilation tools, release 13.1, V13.1.1".
cuda_release=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
cuda_major_version="${cuda_release%%.*}"
if [[ -z "${cuda_major_version}" ]]; then
  echo "Failed to detect CUDA major version from 'nvcc --version' output:" >&2
  nvcc --version >&2
  exit 1
fi
case "${cuda_major_version}" in
  12 | 13) ;;
  *)
    echo "Unsupported CUDA major version '${cuda_major_version}': cuda-stf ships only cu12 and cu13 extras" >&2
    exit 1
    ;;
esac

setup_python_env "${py_version}"

# Locate exactly one wheel matching the given glob under the shared wheelhouse.
find_one_wheel() {
  local glob="$1"
  local wheelhouse="/home/coder/cccl/wheelhouse"
  local wheels
  # Glob-expand into an array. A non-matching glob yields the literal pattern
  # (caught by the existence check below), matching the previous behavior.
  # ${glob} is intentionally unquoted so the shell expands the wildcard.
  # shellcheck disable=SC2086
  mapfile -t wheels < <(printf '%s\n' "${wheelhouse}"/${glob})

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

# Fetch or build the current-branch cuda_cccl and cuda_stf wheels. Both come
# from a single combined producer job so the STF tests exercise the PR's
# cuda-cccl (not a released one from PyPI).
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  # cuda-cccl is uploaded by the combined STF producer under the 'cccl-stf'
  # kind (see ci/build_cuda_stf_combined_python.sh) to avoid colliding with the
  # regular build_py_wheel 'wheel-cccl-...' artifact.
  cccl_artifact_name=$(CCCL_WHEEL_KIND=cccl-stf "$ci_dir/util/workflow/get_wheel_artifact_name.sh")
  "$ci_dir/util/artifacts/download.sh" "${cccl_artifact_name}" /home/coder/cccl/
  stf_artifact_name=$(CCCL_WHEEL_KIND=stf "$ci_dir/util/workflow/get_wheel_artifact_name.sh")
  "$ci_dir/util/artifacts/download.sh" "${stf_artifact_name}" /home/coder/cccl/
else
  "$ci_dir/build_cuda_stf_combined_python.sh" -py-version "${py_version}"
fi

# Install both local wheels in a single resolver invocation so pip binds the
# local cuda_cccl to satisfy cuda-stf's dependency instead of resolving it
# from PyPI.
CUDA_CCCL_WHEEL_PATH="$(find_one_wheel 'cuda_cccl-*.whl')"
CUDA_STF_WHEEL_PATH="$(find_one_wheel 'cuda_stf-*.whl')"
python -m pip install \
    "${CUDA_CCCL_WHEEL_PATH}" \
    "${CUDA_STF_WHEEL_PATH}[test-cu${cuda_major_version}]"

# Run STF tests and examples
cd "/home/coder/cccl/python/cuda_stf/tests/"
python -m pytest -n auto -v stf/
python -m pytest -n 6 -v test_examples.py
