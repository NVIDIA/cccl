#!/usr/bin/env bash
# Test payload: the cuda.compute examples and benchmark smoke tests.
# Invoked by ci/test_cuda_cccl_examples_python.sh, which has already
# put the cuda_cccl wheel in wheelhouse/.
#
# This may run inside the minimal container (see run_in_minimal_container.sh),
# so everything here must work with nothing but Python and the wheel's declared
# pip dependencies.

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
repo_root="$(cd "$ci_dir/.." && pwd)"
# shellcheck source=ci/pyenv_helper.sh
source "$ci_dir/pyenv_helper.sh"

# shellcheck source=ci/util/python/common_arg_parser.sh
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"

# Pin cuda-toolkit to the container's CTK minor and set cuda_version /
# cuda_major_version (-ctk-mode latest opts out). See pyenv_helper.sh.
pin_cuda_toolkit "${ctk_mode}"

# Setup Python environment
setup_python_env "${py_version}"

# Install cuda_cccl, plus CuPy which the cuda.compute examples require, plus
# pytest-benchmark for the host-overhead benchmark smoke test below. (cuda-bench,
# for the throughput smoke, is installed best-effort further down since it does
# not always ship a wheel for the newest Python.)
CUDA_CCCL_WHEEL_PATH="$(ls "${repo_root}"/wheelhouse/cuda_cccl-*.whl)"
ctk_flavor="$(ctk_extra_flavor "${ctk_mode}")"

# CuPy resolves its CUDA libraries (curand, cublas, cufft, ...) through
# cuda-pathfinder at first use, and nothing else in this environment installs
# them -- cuda-cccl only pulls the handful cuda.compute itself needs. On the
# pip-toolkit lanes that leaves e.g. libcurand missing, which the examples that
# call cp.random then fail on; request CuPy's own `ctk` extra so it brings the
# set it needs. The sysctk lanes deliberately use the system toolkit, so they
# stay bare -- pulling the pip wheels there would defeat the lane.
cupy_req="cupy-cuda${cuda_major_version}x"
if [[ "${ctk_flavor}" == "cu" ]]; then
  cupy_req="${cupy_req}[ctk]"
fi

python -m pip install "${CUDA_CCCL_WHEEL_PATH}[test-${ctk_flavor}${cuda_major_version}]" "${cupy_req}" pytest-benchmark

# Run tests for parallel module
cd "${repo_root}/python/cuda_cccl/tests/"
python -m pytest -n 6 test_examples.py

# Smoke-test the host-overhead benchmark harness: run every benchmark case
# exactly once (pass/fail only, no timing) so harness rot fails CI here instead
# of silently surviving until someone runs the perf suite.
cd "${repo_root}/python/cuda_cccl/benchmarks/compute/host/"
python -m pytest -v --benchmark-disable .

# Smoke-test the throughput (nvbench) benchmarks the same way. --profile runs
# each configuration once (no sampling); --quick uses the reduced quick_configs
# axes (one dtype, smallest size) so every benchmark harness still imports,
# registers, launches, and completes. cuda-bench does not always ship a wheel for
# the newest Python, so skip the throughput smoke ONLY for that known no-wheel
# case. Note pip prints "No matching distribution"/"Could not find a version"
# even when the index is unreachable, so check for fetch/network failures first
# and fail on those; any other install error fails the lane too rather than
# silently passing. tee streams pip's output live while capturing it for the
# grep checks below (pipefail keeps pip's exit status, not tee's).
install_log="$(mktemp)"
if python -m pip install "cuda-bench[cu${cuda_major_version}]" pyyaml 2>&1 | tee "${install_log}"; then
  cd "${repo_root}/python/cuda_cccl/benchmarks/compute/"
  python run_benchmarks.py --py --profile --quick
elif grep -qiE "Could not fetch URL|Retrying \(Retry|connection broken|Failed to establish a new connection|Name or service not known|timed out|SSLError|certificate verify failed|ProxyError" "${install_log}"; then
  echo "::error::cuda-bench install failed because pip could not reach the package index (network/DNS/TLS/auth); not skipping." >&2
  exit 1
elif grep -qiE "No matching distribution found for cuda-bench|Could not find a version that satisfies the requirement cuda-bench" "${install_log}"; then
  echo "::warning::cuda-bench has no wheel for Python ${py_version}; skipping the throughput benchmark smoke test."
else
  echo "::error::cuda-bench install failed for an unrecognized reason." >&2
  exit 1
fi
rm -f "${install_log}"
