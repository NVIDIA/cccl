#!/usr/bin/env bash

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$ci_dir/.." && pwd)"
source "$ci_dir/pyenv_helper.sh"

# Parse common arguments
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"
require_py_version "Usage: $0 -py-version <python_version>"

# Pin cuda-toolkit to the container's CTK minor and set cuda_version /
# cuda_major_version (CCCL_PYTHON_TEST_LATEST_CTK=1 opts out). See pyenv_helper.sh.
pin_cuda_toolkit

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

  # Broad thread-safety sweep (pytest-run-parallel): re-run the numba-free
  # functional suite with each test executed concurrently across threads
  # (barrier-synchronized start), stressing the process-wide build cache,
  # single-flight coordination, and the Cython bindings from many threads at
  # once. Complements test_free_threading_stress.py above, which targets specific
  # shared-object scenarios by hand. -n 0 so the threads share one interpreter.
  #
  # --parallel-threads=2 matches CuPy's free-threading CI (the closest GPU
  # precedent); a small fixed count bounds GPU-memory pressure from concurrent
  # kernels and stays reproducible across runners, unlike =auto (the runner's
  # logical-core count).
  #
  # pytest-run-parallel is only used by this sweep, so install it on the 3.14t
  # path rather than for every minimal (e.g. non-free-threaded 3.14) run.
  python -m pip install pytest-run-parallel

  # Fail fast if the interpreter is not actually GIL-free (wrong build /
  # PYTHON_GIL=1): pytest-run-parallel does NOT catch a GIL that is enabled from
  # the start -- it would run threads GIL-serialized and pass vacuously. (A GIL
  # *re-enabled mid-run* by a non-free-threaded import IS caught by the plugin,
  # which is why we do not pass --ignore-gil-enabled.)
  python -c "import sys; assert not sys._is_gil_enabled(), 'GIL is enabled; parallel sweep has no signal'"
  python -m pytest -n 0 -v --parallel-threads=2 compute/test_no_numba.py
fi
