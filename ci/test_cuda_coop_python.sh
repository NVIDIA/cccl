#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$ci_dir/.." && pwd)"

usage="Usage: $0 -py-version <python_version> [-stage <stage>]"

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

needs_cuda_toolkit=false
needs_gpu=false
case "$stage" in
  contracts)
    ;;
  numba-mlir)
    needs_cuda_toolkit=true
    needs_gpu=true
    ;;
  cutlass-host)
    needs_cuda_toolkit=true
    ;;
  cutlass-gpu | cutlass-final-link)
    needs_cuda_toolkit=true
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

if [[ "$needs_cuda_toolkit" == true ]]; then
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
  cutlass-host | cutlass-gpu | cutlass-final-link)
    if [[ "$cuda_major_version" != "13" ]]; then
      echo "Error: CUTLASS stages currently require CUDA 13" >&2
      exit 1
    fi
    python -m pip install "${wheels[0]}[test,cutlass]" torch
    ;;
  *)
    echo "Error: unhandled cuda.coop test stage '$stage'" >&2
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
        "cub/block/block_load.cuh",
        "cub/block/block_store.cuh",
        "cuda/experimental/coop.cuh",
        "cuda/experimental/group.cuh",
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
    python - <<'PY'
import importlib.metadata

import numba_cuda_mlir.cuda as cuda

if not cuda.is_available():
    raise SystemExit("numba-cuda-mlir cannot access an NVIDIA GPU")
print(f"numba-cuda-mlir={importlib.metadata.version('numba-cuda-mlir')}")
PY
    python -m pytest -v \
      backends/numba_mlir/unit/ \
      backends/numba_mlir/compile/ \
      backends/numba_mlir/runtime/
    ;;
  cutlass-host)
    python - <<'PY'
import importlib.metadata

import cutlass.cute  # noqa: F401

print(f"nvidia-cutlass-dsl={importlib.metadata.version('nvidia-cutlass-dsl')}")
PY
    python -m pytest -v \
      backends/cutlass/unit/ \
      backends/cutlass/compile/
    ;;
  cutlass-gpu | cutlass-final-link)
    python - <<'PY'
import importlib.metadata

import cutlass.cute as cute
import torch

if not callable(getattr(cute, "_get_launch_facts", None)):
    raise SystemExit("CUTLASS DSL does not provide launch-facts support")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access an NVIDIA GPU")
print(f"nvidia-cutlass-dsl={importlib.metadata.version('nvidia-cutlass-dsl')}")
PY
    if [[ "$stage" == "cutlass-gpu" ]]; then
      unset CUDA_COOP_CUTLASS_FINAL_LINK_TEST
      python -m pytest -v backends/cutlass/runtime/
    else
      export CUDA_COOP_CUTLASS_FINAL_LINK_TEST=1
      unset CUDA_COOP_CUTLASS_PROVIDER_CCCL_ROOT
      unset CUDA_COOP_CCCL_ROOT
      python -m pytest -v \
        backends/cutlass/runtime/test_group_hierarchy.py::test_local_physical_and_exhaustive_mapped_groups_runtime \
        backends/cutlass/runtime/test_group_hierarchy.py::test_non_exhaustive_mapped_group_membership_runtime \
        backends/cutlass/runtime/test_data_movement.py::test_block_and_warp_data_movement_match_independent_oracles \
        backends/cutlass/runtime/test_data_movement.py::test_fixed_capacity_storage_reaches_block_load_and_store \
        backends/cutlass/runtime/test_reduce_scan.py::test_reduce_scan_group_routes_match_independent_oracles
    fi
    ;;
  *)
    echo "Error: unhandled cuda.coop test stage '$stage'" >&2
    exit 1
    ;;
esac
