#!/usr/bin/env bash
# Test payload: the cuda.cccl.headers suite.
# Invoked by ci/test_cuda_cccl_headers_python.sh, which has already
# put the cuda_cccl wheel in wheelhouse/.
#
# Runs in the minimal container: nothing here may assume more than Python and
# the wheel's declared deps (docs/infrastructure/ci/references/ci_scripts.rst).
# The suite only imports cuda.cccl and asserts on get_include_paths(), so this
# is the lane that most directly tests what the wheel actually ships.

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
repo_root="$(cd "$ci_dir/.." && pwd)"
# shellcheck source=ci/pyenv_helper.sh
source "$ci_dir/pyenv_helper.sh"

python_payload_init "$@"

# Install cuda_cccl. The extra flavor is "cu" (pip-installed toolkit) or "sysctk"
# (system-provided toolkit) depending on the -ctk-mode arg.
CUDA_CCCL_WHEEL_PATH="$(cuda_cccl_wheel_path)"
ctk_flavor="$(ctk_extra_flavor "${ctk_mode}")"
python -m pip install "${CUDA_CCCL_WHEEL_PATH}[test-${ctk_flavor}${cuda_major_version}]"

cd "${repo_root}/python/cuda_cccl/tests/"
python -m pytest -n auto -v headers/
