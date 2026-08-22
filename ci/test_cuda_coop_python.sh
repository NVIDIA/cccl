#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$ci_dir/.." && pwd)"

usage="Usage: $0 -py-version <python_version> [-host-only|-cutlass] [additional options...]"

test_mode="cutlass"
for argument in "$@"; do
  case "$argument" in
    -host-only) test_mode="host" ;;
    -cutlass) test_mode="cutlass" ;;
    *) ;;
  esac
done

# shellcheck source=ci/util/python/common_arg_parser.sh
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"
require_py_version "$usage" || exit 1

# shellcheck source=ci/pyenv_helper.sh
source "$ci_dir/pyenv_helper.sh"

if [[ "$test_mode" == "cutlass" ]]; then
  pin_cuda_toolkit "${ctk_mode}"
  if [[ "${cuda_version}" != "13.3" ]]; then
    echo "Unsupported CUDA version '${cuda_version}': the initial cuda-coop CUTLASS lane requires CUDA 13.3" >&2
    exit 1
  fi
fi

setup_python_env "${py_version}" ".cccl-coop-test-venv"

if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  wheel_artifact_name="$(CCCL_WHEEL_KIND=coop "$ci_dir/util/workflow/get_wheel_artifact_name.sh")"
  "$ci_dir/util/artifacts/download.sh" "$wheel_artifact_name" "$repo_root/"
else
  CCCL_CUDA_COOP_PYENV_READY=1 \
    "$ci_dir/build_cuda_coop_python.sh" -py-version "${py_version}"
fi

mapfile -t wheels < <(find "$repo_root/wheelhouse" -maxdepth 1 -name 'cuda_coop-*.whl' -print | sort)
if [[ ${#wheels[@]} -ne 1 ]]; then
  echo "Error: expected exactly one cuda-coop wheel, found ${#wheels[@]}:" >&2
  printf '  %s\n' "${wheels[@]}" >&2
  exit 1
fi

cd "$repo_root/python/cuda_coop/tests"
if [[ "$test_mode" == "host" ]]; then
  python -m pip install "${wheels[0]}[test]"
  python -m pip check
  python -I - <<'PY'
from pathlib import Path

from cuda import coop
from cuda.coop._headers import resolve_include_paths

assert coop.this_block().kind == "block"
paths = resolve_include_paths(
    start=Path("/tmp/cuda-coop-installed-wheel-probe"),
    required_headers=(
        "cub/block/block_load.cuh",
        "cub/block/block_store.cuh",
        "thrust/detail/raw_pointer_cast.h",
        "cuda/std/cstdint",
    ),
)
assert paths.origin == "cuda-coop wheel header bundle"
PY
  python -m pytest -v contracts/ packaging/
else
  python -m pip install "${wheels[0]}[cutlass,examples,test]"
  python -m pip check
  python -c "import cuda.coop.cutlass"
  python -m pytest -v cutlass_backend/
  python "$repo_root/python/cuda_coop/examples/cutlass/block_load_store.py"
fi
