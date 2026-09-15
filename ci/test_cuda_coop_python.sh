#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$ci_dir/.." && pwd)"

usage="Usage: $0 -py-version <python_version> [-stage contracts|numba-mlir-compile|numba-mlir-runtime]"

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
case "$stage" in
  contracts)
    ;;
  numba-mlir-compile)
    needs_cuda_toolkit=true
    ;;
  numba-mlir-runtime)
    needs_cuda_toolkit=true
    ;;
  *)
    echo "Error: unknown cuda.coop test stage '$stage'" >&2
    echo "$usage" >&2
    exit 1
    ;;
esac

if [[ "$stage" == "numba-mlir-compile" ]]; then
  # Hide every device for the complete compiler-contract stage, including
  # import and installed-wheel isolation probes.
  export CUDA_VISIBLE_DEVICES=""
fi

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

case "$stage" in
  contracts)
    python -m pip install "${wheels[0]}[test]"
    ;;
  numba-mlir-compile | numba-mlir-runtime)
    python -m pip install \
      "${wheels[0]}[test,numba-cuda-mlir-cu${cuda_major_version}]"
    ;;
  *)
    echo "Error: unhandled cuda.coop test stage '$stage'" >&2
    exit 1
    ;;
esac

python -m pip check
python -I - <<'PY'
import importlib.metadata
import sys
from pathlib import Path

import cuda
from cuda import coop
from cuda.coop._headers import resolve_include_paths

distribution = importlib.metadata.distribution("cuda-coop")
expected_root = Path(
    distribution.locate_file("cuda/coop/__init__.py")
).resolve()
assert Path(coop.__file__).resolve() == expected_root
assert getattr(cuda, "__file__", None) is None
assert coop.this_block().kind == "block"

paths = resolve_include_paths(
    start=Path(sys.prefix),
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

tests_root="$repo_root/python/cuda_coop/tests"

case "$stage" in
  contracts)
    cd "$tests_root"
    python -m pytest -v contracts/ packaging/
    ;;
  numba-mlir-compile)
    # The compile contract is deliberately GPU-free. Tests may replace only
    # the backend's current-device query with a fixed compute capability; NVRTC
    # and nvJitLink remain real.
    python -I - <<'PY'
import importlib.metadata

from numba_cuda_mlir import cuda

if cuda.is_available():
    raise SystemExit("GPU-hidden cuda.coop compile stage can access a CUDA device")
print(f"numba-cuda-mlir={importlib.metadata.version('numba-cuda-mlir')}")
PY
    cd "$tests_root"
    python -m pytest -v \
      backends/numba_mlir/unit/ \
      backends/numba_mlir/compile/
    ;;
  numba-mlir-runtime)
    python -I - <<'PY'
import importlib.metadata

from numba_cuda_mlir import cuda

if not cuda.is_available():
    raise SystemExit("numba-cuda-mlir cannot access an NVIDIA GPU")
print(f"numba-cuda-mlir={importlib.metadata.version('numba-cuda-mlir')}")
PY
    nvidia-smi \
      --query-gpu=name,compute_cap,driver_version \
      --format=csv,noheader
    cd "$tests_root"
    python -m pytest -v backends/numba_mlir/runtime/

    mapfile -t examples < <(
      find "$repo_root/python/cuda_coop/examples/numba_mlir" \
        -maxdepth 1 -name '*.py' ! -name '__init__.py' -print | sort
    )
    if [[ ${#examples[@]} -eq 0 ]]; then
      echo "Error: no cuda.coop Numba-CUDA-MLIR examples were found" >&2
      exit 1
    fi
    for example in "${examples[@]}"; do
      python -I "$example"
    done
    ;;
  *)
    echo "Error: unhandled cuda.coop test stage '$stage'" >&2
    exit 1
    ;;
esac
