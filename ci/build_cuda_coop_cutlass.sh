#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build the standalone sdist, then its universal wheel from the extracted sdist.
set -euo pipefail
ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$ci_dir/.." && pwd)"
usage="Usage: $0 -py-version <python_version>"
# shellcheck source=ci/util/python/common_arg_parser.sh
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"
require_py_version "$usage" || exit 1
# shellcheck source=ci/pyenv_helper.sh
source "$ci_dir/pyenv_helper.sh"
if [[ "${CCCL_CUDA_COOP_CUTLASS_PYENV_READY:-0}" != "1" ]]; then
  setup_python_env "${py_version}" ".cccl-coop-cutlass-build-venv"
fi
python -m pip install build
artifacts_dir="${CCCL_CUDA_COOP_CUTLASS_ARTIFACT_DIR:-$repo_root/wheelhouse}"
mkdir -p "$artifacts_dir"
python -m build --outdir "$artifacts_dir" "$repo_root/python/cuda_coop_cutlass"
mapfile -t wheels < <(find "$artifacts_dir" -maxdepth 1 -name 'cuda_coop_cutlass-*.whl' -print | sort)
mapfile -t sdists < <(find "$artifacts_dir" -maxdepth 1 -name 'cuda_coop_cutlass-*.tar.gz' -print | sort)
if [[ ${#wheels[@]} -ne 1 || ${#sdists[@]} -ne 1 ]]; then
  echo "Error: expected exactly one cuda-coop-cutlass wheel and sdist in $artifacts_dir" >&2
  exit 1
fi
python "$ci_dir/validate_cuda_coop_cutlass_artifacts.py" "${wheels[0]}" "${sdists[0]}"
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  artifact_name="$(CCCL_WHEEL_KIND=coop-cutlass "$ci_dir/util/workflow/get_wheel_artifact_name.sh")"
  "$ci_dir/util/artifacts/upload.sh" "$artifact_name" 'wheelhouse/cuda_coop_cutlass-.*\.(whl|tar\.gz)'
fi
