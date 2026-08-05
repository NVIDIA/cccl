#!/usr/bin/env bash

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$ci_dir/.." && pwd)"

# shellcheck source=ci/pyenv_helper.sh
source "$ci_dir/pyenv_helper.sh"
# shellcheck source=ci/util/python/common_arg_parser.sh
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"
pin_cuda_toolkit "${ctk_mode}"
setup_python_env "${py_version}"

if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  wheel_artifact_name=$("$ci_dir/util/workflow/get_wheel_artifact_name.sh")
  "$ci_dir/util/artifacts/download.sh" "${wheel_artifact_name}" "${repo_root}/"
else
  "$ci_dir/build_cuda_cccl_python.sh" -py-version "${py_version}"
fi

# Install both coordinated wheels from explicit paths. Index access remains
# available for the selected CUDA extra's third-party dependencies.
wheelhouse_dir="${repo_root}/wheelhouse"
mapfile -t cuda_cccl_wheels < <(
  find "${wheelhouse_dir}" -maxdepth 1 -type f -name 'cuda_cccl-*.whl' -print | sort
)
mapfile -t cuda_compute_wheels < <(
  find "${wheelhouse_dir}" -maxdepth 1 -type f -name 'cuda_compute-*.whl' -print | sort
)
if [[ "${#cuda_cccl_wheels[@]}" -ne 1 ]]; then
  echo "Expected exactly one cuda-cccl wheel in ${wheelhouse_dir}; found ${#cuda_cccl_wheels[@]}." >&2
  exit 1
fi
if [[ "${#cuda_compute_wheels[@]}" -ne 1 ]]; then
  echo "Expected exactly one cuda-compute wheel in ${wheelhouse_dir}; found ${#cuda_compute_wheels[@]}." >&2
  exit 1
fi
CUDA_CCCL_WHEEL_PATH="${cuda_cccl_wheels[0]}"
CUDA_COMPUTE_WHEEL_PATH="${cuda_compute_wheels[0]}"
ctk_flavor="$(ctk_extra_flavor "${ctk_mode}")"
python -m pip install --find-links "${wheelhouse_dir}" \
  "${CUDA_CCCL_WHEEL_PATH}[minimal-${ctk_flavor}${cuda_major_version}]" \
  "${CUDA_COMPUTE_WHEEL_PATH}"
python -m pip check
python - <<'PY'
import importlib.metadata
import importlib.util

import cuda.compute

assert importlib.metadata.version("cuda-cccl") == importlib.metadata.version(
    "cuda-compute"
)
assert importlib.util.find_spec("cuda.cccl") is None
PY
