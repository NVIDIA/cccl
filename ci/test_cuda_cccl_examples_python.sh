#!/usr/bin/env bash

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ci_dir/pyenv_helper.sh"

# Parse common arguments
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"
# Pin cuda-toolkit to the container's CTK minor and set cuda_version /
# cuda_major_version (CCCL_PYTHON_TEST_LATEST_CTK=1 opts out). See pyenv_helper.sh.
pin_cuda_toolkit

# Setup Python environment
setup_python_env "${py_version}"

# Fetch or build the cuda_cccl wheel:
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  wheel_artifact_name=$("$ci_dir/util/workflow/get_wheel_artifact_name.sh")
  "$ci_dir/util/artifacts/download.sh" "${wheel_artifact_name}" /home/coder/cccl/
else
  "$ci_dir/build_cuda_cccl_python.sh" -py-version "${py_version}"
fi

# Install cuda_cccl, plus CuPy which the cuda.compute examples require, plus
# pytest-benchmark for the host-overhead benchmark smoke test below. (cuda-bench,
# for the throughput smoke, is installed best-effort further down since it does
# not always ship a wheel for the newest Python.)
CUDA_CCCL_WHEEL_PATH="$(ls /home/coder/cccl/wheelhouse/cuda_cccl-*.whl)"
python -m pip install "${CUDA_CCCL_WHEEL_PATH}[test-cu${cuda_major_version}]" "cupy-cuda${cuda_major_version}x" pytest-benchmark

# Run tests for parallel module
cd "/home/coder/cccl/python/cuda_cccl/tests/"
python -m pytest -n 6 test_examples.py

# Smoke-test the host-overhead benchmark harness: run every benchmark case
# exactly once (pass/fail only, no timing) so harness rot fails CI here instead
# of silently surviving until someone runs the perf suite.
cd "/home/coder/cccl/python/cuda_cccl/benchmarks/compute/host/"
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
  cd "/home/coder/cccl/python/cuda_cccl/benchmarks/compute/"
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
