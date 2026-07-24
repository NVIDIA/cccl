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
else
  # Clear any inherited constraint so this lane truly tests the latest minor.
  unset PIP_CONSTRAINT
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
# case; any other pip failure (index outage, dependency conflict, bad metadata)
# fails the lane rather than silently passing.
if install_log=$(python -m pip install "cuda-bench[cu${cuda_major_version}]" pyyaml 2>&1); then
  echo "${install_log}"
  cd "/home/coder/cccl/python/cuda_cccl/benchmarks/compute/"
  python run_benchmarks.py --py --profile --quick
elif grep -qiE "No matching distribution found for cuda-bench|Could not find a version that satisfies the requirement cuda-bench" <<<"${install_log}"; then
  echo "${install_log}"
  echo "::warning::cuda-bench has no wheel for Python ${py_version}; skipping the throughput benchmark smoke test."
else
  echo "${install_log}" >&2
  echo "::error::cuda-bench install failed for a reason other than a missing wheel." >&2
  exit 1
fi
