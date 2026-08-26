#!/usr/bin/env bash
# Test payload: the cuda.compute pytest suite.
# Invoked by ci/test_cuda_compute_python.sh, which has already
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

# Install cuda_cccl. The extra flavor is "cu" (pip-installed toolkit) or "sysctk"
# (system-provided toolkit) depending on the -ctk-mode arg.
CUDA_CCCL_WHEEL_PATH="$(ls "${repo_root}"/wheelhouse/cuda_cccl-*.whl)"
ctk_flavor="$(ctk_extra_flavor "${ctk_mode}")"
python -m pip install "${CUDA_CCCL_WHEEL_PATH}[test-${ctk_flavor}${cuda_major_version}]"

# Run tests for compute module.
# On the v2 (HostJIT) backend, abort on first failure — the suite is still
# stabilizing and a single early failure is enough signal to investigate
# without scrolling through hundreds of subsequent passes.
pytest_extra=()
if [[ "${CCCL_PYTHON_USE_V2:-}" =~ ^(1|true|TRUE|on|ON)$ ]]; then
  pytest_extra+=(-x)
fi

cd "${repo_root}/python/cuda_cccl/tests/"
if [[ "${CCCL_PYTHON_USE_V2:-}" =~ ^(1|true|TRUE|on|ON)$ ]]; then
  # The test isolates itself in a fresh subprocess (LLVM initialization is
  # process-wide and only cold once), but it carries the free_threading marker,
  # so it must be selected by node-id here or the sweeps below never run it.
  python -m pytest "${pytest_extra[@]}" -n 0 -v \
    compute/test_free_threading_stress.py::test_v2_concurrent_cold_llvm_initialization
fi
python -m pytest "${pytest_extra[@]}" -n 6 -v compute/ -m "not large and not free_threading"
python -m pytest "${pytest_extra[@]}" -n 0 -v compute/ -m "large and not free_threading"

# The bfloat16 tests require ml_dtypes (the NumPy bfloat16 extension dtype),
# which is deliberately not part of the test extras so that the sweeps above
# run in an environment matching a user's default install (where the bfloat16
# tests skip themselves). Install it last and run those tests explicitly.
python -m pip install ml_dtypes
python -m pytest "${pytest_extra[@]}" -n 6 -v compute/test_bfloat16.py
