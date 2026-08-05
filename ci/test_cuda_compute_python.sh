#!/usr/bin/env bash

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$ci_dir/.." && pwd)"
source "$ci_dir/pyenv_helper.sh"

# Parse common arguments
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"
# Pin cuda-toolkit to the container's CTK minor and set cuda_version /
# cuda_major_version (-ctk-mode latest opts out). See pyenv_helper.sh.
pin_cuda_toolkit "${ctk_mode}"

# Setup Python environment
setup_python_env "${py_version}"

# Fetch or build the coordinated wheelhouse:
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  wheel_artifact_name=$("$ci_dir/util/workflow/get_wheel_artifact_name.sh")
  "$ci_dir/util/artifacts/download.sh" "${wheel_artifact_name}" "${repo_root}/"
else
  "$ci_dir/build_cuda_cccl_python.sh" -py-version "${py_version}"
fi

# Install cuda-compute directly. The extra flavor is "cu" (pip-installed
# toolkit) or "sysctk"
# (system-provided toolkit) depending on the -ctk-mode arg.
wheelhouse_dir="${repo_root}/wheelhouse"
mapfile -t cuda_compute_wheels < <(
  find "${wheelhouse_dir}" -maxdepth 1 -type f -name 'cuda_compute-*.whl' -print | sort
)
if [[ "${#cuda_compute_wheels[@]}" -ne 1 ]]; then
  echo "Expected exactly one cuda-compute wheel in ${wheelhouse_dir}; found ${#cuda_compute_wheels[@]}." >&2
  exit 1
fi
CUDA_COMPUTE_WHEEL_PATH="${cuda_compute_wheels[0]}"
ctk_flavor="$(ctk_extra_flavor "${ctk_mode}")"
python -m pip install --find-links "${wheelhouse_dir}" \
  "${CUDA_COMPUTE_WHEEL_PATH}[test-${ctk_flavor}${cuda_major_version}]"
python -m pip check
python - <<'PY'
import importlib.metadata

try:
    importlib.metadata.distribution("cuda-cccl")
except importlib.metadata.PackageNotFoundError:
    pass
else:
    raise AssertionError("direct cuda-compute install must not install cuda-cccl")
PY

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
