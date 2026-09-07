#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$ci_dir/.." && pwd)"

usage="Usage: $0 -py-version <python_version> [-stage contracts|numba-mlir]"

# shellcheck source=ci/util/python/common_arg_parser.sh
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"
require_py_version "$usage" || exit 1

stage="contracts"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -stage | --stage)
      if [[ $# -lt 2 || -z "$2" ]]; then
        echo "Error: $1 requires a value" >&2
        exit 1
      fi
      stage="$2"
      shift 2
      ;;
    -stage=* | --stage=*)
      stage="${1#*=}"
      if [[ -z "$stage" ]]; then
        echo "Error: $1 requires a value" >&2
        exit 1
      fi
      shift
      ;;
    -py-version | -ctk-mode)
      if [[ $# -lt 2 || -z "$2" ]]; then
        echo "Error: $1 requires a value" >&2
        exit 1
      fi
      shift 2
      ;;
    -py-version=* | -ctk-mode=*)
      shift
      ;;
    *)
      echo "Error: unknown argument '$1'" >&2
      echo "$usage" >&2
      exit 1
      ;;
  esac
done

case "$stage" in
  contracts)
    needs_gpu=false
    ;;
  numba-mlir)
    needs_gpu=true
    ;;
  *)
    echo "Error: unknown cuda.coop test stage '$stage'" >&2
    echo "$usage" >&2
    exit 1
    ;;
esac

# shellcheck source=ci/pyenv_helper.sh
source "$ci_dir/pyenv_helper.sh"

if [[ "$stage" == "numba-mlir" ]]; then
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "Error: cuda.coop stage '$stage' requires nvcc on PATH" >&2
    exit 1
  fi
  pin_cuda_toolkit "${ctk_mode}"
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
case "$stage" in
  contracts)
    python -m pip install "${wheels[0]}[test]"
    ;;
  numba-mlir)
    python -m pip install \
      "${wheels[0]}[test,numba-cuda-mlir-cu${cuda_major_version}]"
    ;;
  *)
    echo "Error: unsupported cuda-coop test stage: ${stage}" >&2
    exit 1
    ;;
esac

python -m pip check
python -I - <<'PY'
from pathlib import Path

from cuda import coop
from cuda.coop._headers import resolve_include_paths

assert coop.this_block().kind == "block"
paths = resolve_include_paths(
    start=Path("/tmp/cuda-coop-installed-wheel-probe"),
    required_headers=(
        "cub/block/block_reduce.cuh",
        "thrust/detail/raw_pointer_cast.h",
        "cuda/std/cstdint",
    ),
)
assert paths.origin == "cuda-coop wheel header bundle"
PY

if [[ "$needs_gpu" == true ]]; then
  nvidia-smi \
    --query-gpu=name,compute_cap,driver_version \
    --format=csv,noheader
fi

case "$stage" in
  contracts)
    python -m pytest -v contracts/ packaging/
    ;;
  numba-mlir)
    python -I - <<'PY'
import importlib.metadata
from pathlib import Path

from numba_cuda_mlir import cuda as mlir_cuda

import cuda.coop.numba_mlir as coop
from cuda.coop.numba_mlir._compiler import _group_planner
from cuda.coop.numba_mlir._lowering import _reduce

distribution = importlib.metadata.distribution("cuda-coop")
expected_backend = Path(
    distribution.locate_file("cuda/coop/numba_mlir/__init__.py")
).resolve()
expected_group_planner = Path(
    distribution.locate_file("cuda/coop/numba_mlir/_compiler/_group_planner.py")
).resolve()
expected_reduce_lowering = Path(
    distribution.locate_file("cuda/coop/numba_mlir/_lowering/_reduce.py")
).resolve()
assert Path(coop.__file__).resolve() == expected_backend
assert Path(_group_planner.__file__).resolve() == expected_group_planner
assert Path(_reduce.__file__).resolve() == expected_reduce_lowering
if not mlir_cuda.is_available():
    raise SystemExit("numba-cuda-mlir cannot access an NVIDIA GPU")
print(f"numba-cuda-mlir={importlib.metadata.version('numba-cuda-mlir')}")
PY
    python -m pytest -v \
      backends/numba_mlir/unit/ \
      backends/numba_mlir/compile/ \
      backends/numba_mlir/runtime/
    python ../examples/numba_mlir/block_reduce.py
    ;;
  *)
    echo "Error: unsupported cuda-coop test stage: ${stage}" >&2
    exit 1
    ;;
esac
