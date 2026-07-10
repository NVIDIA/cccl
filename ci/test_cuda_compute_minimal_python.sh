#!/usr/bin/env bash

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$ci_dir/.." && pwd)"
source "$ci_dir/pyenv_helper.sh"

# Parse common arguments
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"
require_py_version "Usage: $0 -py-version <python_version>"

cuda_major_version=$(nvcc --version | grep release | awk '{print $6}' | tr -d ',' | cut -d '.' -f 1 | cut -d 'V' -f 2)

# Setup Python environment
setup_python_env "${py_version}"

# Fetch or build the cuda_cccl wheel:
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  wheel_artifact_name=$("$ci_dir/util/workflow/get_wheel_artifact_name.sh")
  "$ci_dir/util/artifacts/download.sh" "${wheel_artifact_name}" "${repo_root}/"
  wheelhouse_dir="${repo_root}/wheelhouse"
else
  "$ci_dir/build_cuda_cccl_python.sh" -py-version "${py_version}"
  wheelhouse_dir="${repo_root}/wheelhouse"
fi

# Install cuda_cccl with the minimal CUDA extra. This intentionally avoids the
# full cu* extras because those pull in numba/numba-cuda.
CUDA_CCCL_WHEEL_PATH="$(ls "${wheelhouse_dir}"/cuda_cccl-*.whl)"
python -m pip install "${CUDA_CCCL_WHEEL_PATH}[minimal-cu${cuda_major_version}]"
python -m pip install pytest pytest-xdist

cd "${repo_root}/python/cuda_cccl/tests/"
python -m pytest -n 6 -v compute/test_no_numba.py
if [[ "${py_version}" == "3.14t" ]]; then
  # Select only tests that support the minimal extra so pytest does not collect
  # tests that import numba-cuda and re-enable the GIL. These tests provide their
  # own worker threads, so keep pytest itself in a single process.
  # The serialization node-ids are module-skipped on the v2 backend today and
  # will start running there automatically once v2 gains serialization support.
  python -m pytest -n 0 -v \
    compute/test_free_threading_stress.py \
    compute/test_multi_cc_serialization.py::test_aot_build_result_load_failure_is_shared_and_retryable \
    compute/test_multi_cc_serialization.py::test_aot_serialization_waits_for_canonical_first_load
fi
