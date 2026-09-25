#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$ci_dir/.." && pwd)"

usage="Usage: $0 -py-version <python_version> [additional options...]"

# shellcheck source=ci/util/python/common_arg_parser.sh
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"
require_py_version "$usage" || exit 1

# shellcheck source=ci/pyenv_helper.sh
source "$ci_dir/pyenv_helper.sh"
if [[ "${CCCL_CUDA_COOP_PYENV_READY:-0}" != "1" ]]; then
  setup_python_env "${py_version}" ".cccl-coop-build-venv"
fi

python -m pip install build

cd "$repo_root"
mkdir -p wheelhouse
rm -f wheelhouse/cuda_coop-*.whl
python -m build --wheel --outdir wheelhouse python/cuda_coop

mapfile -t wheels < <(find wheelhouse -maxdepth 1 -name 'cuda_coop-*.whl' -print | sort)
if [[ ${#wheels[@]} -ne 1 ]]; then
  echo "Error: expected exactly one cuda-coop wheel, found ${#wheels[@]}:" >&2
  printf '  %s\n' "${wheels[@]}" >&2
  exit 1
fi

if [[ "${wheels[0]}" != *-py3-none-any.whl ]]; then
  echo "Error: cuda-coop must produce a universal py3-none-any wheel: ${wheels[0]}" >&2
  exit 1
fi

python "$ci_dir/validate_cuda_coop_wheel.py" "${wheels[0]}"

echo "Built cuda-coop wheel: ${wheels[0]}"

if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  wheel_artifact_name="$(CCCL_WHEEL_KIND=coop "$ci_dir/util/workflow/get_wheel_artifact_name.sh")"
  "$ci_dir/util/artifacts/upload.sh" "$wheel_artifact_name" 'wheelhouse/cuda_coop-.*\.whl'
fi
