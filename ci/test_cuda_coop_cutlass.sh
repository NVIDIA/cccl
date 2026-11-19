#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Qualify installed artifacts before allocating a GPU runtime runner.
set -euo pipefail
ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$ci_dir/.." && pwd)"
usage="Usage: $0 -py-version <python_version> [-stage host|compile|runtime] [-suite pr|smoke|full]"
# shellcheck source=ci/util/python/common_arg_parser.sh
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"
require_py_version "$usage" || exit 1
stage=compile
suite=
while [[ $# -gt 0 ]]; do
  case "$1" in
    -stage | --stage)
      [[ $# -ge 2 && -n "$2" ]] || { echo "$usage" >&2; exit 1; }
      stage="$2"; shift 2 ;;
    -stage=* | --stage=*) stage="${1#*=}"; shift ;;
    -suite | --suite)
      [[ $# -ge 2 && -n "$2" ]] || { echo "$usage" >&2; exit 1; }
      suite="$2"; shift 2 ;;
    -suite=* | --suite=*)
      suite="${1#*=}"; shift
      [[ -n "$suite" ]] || { echo "$usage" >&2; exit 1; } ;;
    -py-version | -ctk-mode) shift 2 ;;
    -py-version=* | -ctk-mode=*) shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done
case "$stage:$suite" in
  host: | compile:) export CUDA_VISIBLE_DEVICES="" ;;
  runtime: | runtime:full | runtime:pr | runtime:smoke) ;;
  *) echo "$usage" >&2; exit 1 ;;
esac
# shellcheck source=ci/pyenv_helper.sh
source "$ci_dir/pyenv_helper.sh"
if [[ "$stage" != host ]]; then
  command -v nvcc >/dev/null || { echo "nvcc is required for $stage" >&2; exit 1; }
  pin_cuda_toolkit "${ctk_mode}"
fi
setup_python_env "${py_version}" ".cccl-coop-cutlass-test-venv"
if [[ -n "${GITHUB_ACTIONS:-}" && "$stage" == runtime ]]; then
  artifact_name="$(CCCL_WHEEL_KIND=coop-cutlass "$ci_dir/util/workflow/get_wheel_artifact_name.sh")"
  "$ci_dir/util/artifacts/download.sh" "$artifact_name" "$repo_root/"
else
  CUDA_COOP_CUTLASS_PYENV_READY=1 "$ci_dir/build_cuda_coop_cutlass.sh" -py-version "${py_version}"
fi
mapfile -t wheels < <(find "$repo_root/wheelhouse" -maxdepth 1 -name 'cuda_coop_cutlass-*.whl' -print | sort)
[[ ${#wheels[@]} -eq 1 ]] || { echo "Expected exactly one cuda-coop-cutlass wheel" >&2; exit 1; }
python -m pip install "${wheels[0]}[test]"
if [[ "$stage" != host ]]; then
  python -m pip install -r "$ci_dir/cuda_coop_cutlass_requirements.txt"
fi
python -m pip check
# Isolated mode and disabled PYTHONPATH prevent the checkout from masking defects.
unset PYTHONPATH
python -I - <<'PY'
import importlib.metadata
import sys
from pathlib import Path

import cuda
import cuda.coop
import cuda.coop.cutlass as coop
from cuda.coop.cutlass._headers import resolve_include_paths

package = importlib.metadata.distribution("cuda-coop-cutlass")
assert Path(coop.__file__).resolve() == Path(
    package.locate_file("cuda/coop/cutlass/__init__.py")
).resolve()
assert getattr(cuda, "__file__", None) is None
assert getattr(cuda.coop, "__file__", None) is None
paths = resolve_include_paths(
    start=Path(sys.prefix),
    required_headers=("cub/block/block_load.cuh", "cuda/std/cstdint", "nv/target"),
)
assert paths.origin == "cuda-coop-cutlass wheel header bundle"
print("nvidia-cutlass-dsl=" + importlib.metadata.version("nvidia-cutlass-dsl"))
PY
package_root="$repo_root/python/cuda_coop_cutlass"
cd "$package_root/tests"
case "$stage" in
  host | compile)
    python -I -m pytest -v -m 'not gpu' test_*.py packaging/ unit/
    if [[ "$stage" == compile ]]; then
      for route in 0 1; do
        CUTE_DSL_USE_EXTENSION_COMPILER="$route" python -I -m pytest -v -m 'not gpu' compile/
      done
    fi
    ;;
  runtime)
    python -I - <<'PY'
from cuda.pathfinder import find_nvidia_binary_utility

if find_nvidia_binary_utility("cuobjdump") is None:
    raise SystemExit("cuobjdump is required for final-cubin checks")
PY
    nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
    case "$suite" in
      pr) gpu_tests=(-m 'gpu and not weekly' compile/ runtime/) ;;
      smoke)
        # Exercise the documented operations, final cubins and runtime regressions.
        gpu_tests=(-m gpu
          compile/test_compiler_lifecycle.py
          runtime/test_*_documentation.py
          runtime/test_convergence.py
          runtime/test_primitive_bundle.py
          runtime/test_reserve.py
          runtime/test_block_algorithms.py::test_final_cubin_storage_contract
          runtime/test_block_load_store.py::test_final_cubin_eliminates_direct_providers_and_scratch
          runtime/test_block_load_store.py::test_runtime_load_default_and_offset_change_between_launches
          runtime/test_block_load_store.py::test_unsigned_memory_behind_pointer_arguments
          runtime/test_device_operator.py::test_final_cubin_inlines_provider_and_callback
          runtime/test_device_operator.py::test_same_name_and_signature_keep_distinct_bodies
          runtime/test_exchange.py::test_final_cubin
          runtime/test_group_hierarchy.py::test_mapped_query_cubin
          runtime/test_histogram.py::test_final_cubin
          runtime/test_logical_warp_load_store.py::test_final_cubin
          runtime/test_merge_sort.py::test_final_cubin
          runtime/test_neighbors.py::test_final_cubin
          runtime/test_radix_controls.py::test_final_cubin
          runtime/test_reduce.py::test_final_cubin
          runtime/test_reduce_batched_codegen.py::test_final_cubin
          runtime/test_run_length_controls.py::test_final_cubin
          runtime/test_scan.py::test_final_cubin
          runtime/test_shuffle.py::test_final_cubin
          runtime/test_topk_controls.py::test_final_cubin
          runtime/test_warp_load_store.py::test_final_warp_cubin_has_no_block_barrier
        )
        ;;
      *) gpu_tests=(-m gpu compile/ runtime/) ;;
    esac
    for route in 0 1; do
      CUTE_DSL_USE_EXTENSION_COMPILER="$route" python -I -m pytest -v -n 4 --max-worker-restart=0 "${gpu_tests[@]}"
    done
    # The runtime pytest suite covers TMEM on SM100/SM103 and skips it on
    # other GPUs; its standalone CLI deliberately rejects unsupported devices.
    mapfile -t examples < <(find "$package_root/examples" -maxdepth 1 -name '*.py' ! -name '__init__.py' ! -name '_*' ! -name 'tensor_memory_topk.py' -print | sort)
    [[ ${#examples[@]} -gt 0 ]] || { echo "No CUTLASS examples found" >&2; exit 1; }
    for example in "${examples[@]}"; do
      python -I "$example"
    done
    ;;
  *) echo "Unhandled stage: $stage" >&2; exit 1 ;;
esac
