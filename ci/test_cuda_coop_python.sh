#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Test the installed wheel in one stage: host contracts, device compilation,
# or GPU execution. The compiler stage hides devices to check that providers
# can compile with an explicit target and no running kernel.

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$ci_dir/.." && pwd)"

usage="Usage: $0 -py-version <python_version> [-stage contracts|numba-mlir-compile|numba-mlir-runtime|cutlass-compile|cutlass-runtime|coexistence] [-compute-sanitizer-racecheck]"

# shellcheck source=ci/util/python/common_arg_parser.sh
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"
require_py_version "$usage" || exit 1

stage="contracts"
racecheck=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    -compute-sanitizer-racecheck)
      racecheck=true
      shift
      ;;
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
  numba-mlir-compile | numba-mlir-runtime | cutlass-compile | cutlass-runtime | coexistence)
    needs_cuda_toolkit=true
    ;;
  *)
    echo "Error: unknown cuda.coop test stage '$stage'" >&2
    echo "$usage" >&2
    exit 1
    ;;
esac

if [[ "$racecheck" == true ]]; then
  if [[ "$stage" != "numba-mlir-runtime" ]]; then
    echo "Error: -compute-sanitizer-racecheck requires -stage numba-mlir-runtime" >&2
    exit 1
  fi
  if ! command -v compute-sanitizer >/dev/null 2>&1; then
    echo "Error: racecheck qualification requires compute-sanitizer on PATH" >&2
    exit 1
  fi
  # Pytest runs these instrumented cases serially, including their negative
  # controls. A missing sanitizer must fail this lane rather than skip it.
  export CUDA_COOP_RUN_RACECHECK=1
fi

if [[ "$stage" == "numba-mlir-compile" || "$stage" == "cutlass-compile" || "$stage" == "coexistence" ]]; then
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

wheel_kind="coop"
if [[ "$stage" == "cutlass-compile" || "$stage" == "cutlass-runtime" ]]; then
  # The CPU compiler producer qualifies its own wheel before any GPU job.
  wheel_kind="coop-cutlass"
fi

if [[ -n "${GITHUB_ACTIONS:-}" && "$stage" != "cutlass-compile" ]]; then
  wheel_artifact_name="$(CCCL_WHEEL_KIND="$wheel_kind" "$ci_dir/util/workflow/get_wheel_artifact_name.sh")"
  "$ci_dir/util/artifacts/download.sh" "$wheel_artifact_name" "$repo_root/"
else
  CCCL_CUDA_COOP_PYENV_READY=1 CCCL_WHEEL_KIND="$wheel_kind" \
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
  numba-mlir-compile | numba-mlir-runtime | coexistence)
    python -m pip install \
      "${wheels[0]}[test,numba-cuda-mlir-cu${cuda_major_version}]"
    ;;
  cutlass-compile | cutlass-runtime)
    python -m pip install "${wheels[0]}[test]"
    ;;
  *)
    echo "Error: unhandled cuda.coop test stage '$stage'" >&2
    exit 1
    ;;
esac

case "$stage" in
  cutlass-compile | cutlass-runtime | coexistence)
    # A runtime lacking external LTO-IR
    # support must fail compilation; import-only success is not qualification.
    python -m pip install -r "$ci_dir/cuda_coop_cutlass_requirements.txt"
    python -I - <<'PY'
import importlib.metadata

import cuda.coop.cutlass

print(f"nvidia-cutlass-dsl={importlib.metadata.version('nvidia-cutlass-dsl')}")
PY
    ;;
  *)
    ;;
esac

python -m pip check
# Isolated mode prevents the checkout or PYTHONPATH from hiding wheel defects.
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
        "thrust/detail/raw_pointer_cast.h",
        "cuda/std/cstdint",
    ),
)
assert paths.origin == "cuda-coop wheel header bundle"
PY

tests_root="$repo_root/python/cuda_coop/tests"

case "$stage" in
  contracts)
    # Check shared helpers without requiring optional compiler backends.
    python -m pyright --warnings --project "$repo_root/python/cuda_coop/tests/typing/pyright-unused.json" \
      --pythonpath "$(command -v python)"
    cd "$tests_root"
    python -m pytest -v test_*.py contracts/ packaging/
    ;;
  numba-mlir-compile)
    python -m pyright --project "$repo_root/python/cuda_coop" \
      --pythonpath "$(command -v python)" --warnings
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
  cutlass-compile)
    # Some lifecycle tests live in compile/ but launch kernels. Markers keep
    # those on the runtime runner while explicit-target compilation stays CPU-only.
    cd "$tests_root"
    python -m pytest -v -m 'not gpu' \
      backends/cutlass/unit/ backends/cutlass/compile/
    ;;
  coexistence)
    # Both optional compilers are installed for import-order and registration
    # checks. These probes do not allocate a GPU or compile a kernel.
    cd "$tests_root"
    python -m pytest -v \
      backends/cutlass/unit/test_registration.py \
      backends/numba_mlir/unit/test_runtime_activation.py
    ;;
  cutlass-runtime)
    python -I - <<'PY'
import cutlass

cutlass.cuda.initialize_cuda_context()
PY
    nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
    cd "$tests_root"
    python -m pytest -v -m gpu \
      backends/cutlass/compile/ backends/cutlass/runtime/
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
      case "${example##*/}" in
        nvmath_fft.py | nvmath_fft_shared.py | \
          nvmath_nvshmem.py | nvmath_nvshmem_shared.py)
          # The examples README documents their nvmath/NVSHMEM environments.
          echo "Skipping optional device-library example: ${example##*/}"
          ;;
        source_dumps.py)
          for source_dump_mode in direct transpose scan; do
            python -I "$example" "$source_dump_mode"
          done
          ;;
        *)
          python -I "$example"
          ;;
      esac
    done
    ;;
  *)
    echo "Error: unhandled cuda.coop test stage '$stage'" >&2
    exit 1
    ;;
esac
