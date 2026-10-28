#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Qualify installed artifacts before allocating a GPU runtime runner.
set -euo pipefail
ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$ci_dir/.." && pwd)"
usage="Usage: $0 -py-version <python_version> [-stage host|compile|runtime]"
# shellcheck source=ci/util/python/common_arg_parser.sh
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"
require_py_version "$usage" || exit 1
stage=compile
while [[ $# -gt 0 ]]; do
  case "$1" in
    -stage | --stage)
      [[ $# -ge 2 && -n "$2" ]] || { echo "$usage" >&2; exit 1; }
      stage="$2"; shift 2 ;;
    -stage=* | --stage=*) stage="${1#*=}"; shift ;;
    -py-version | -ctk-mode) shift 2 ;;
    -py-version=* | -ctk-mode=*) shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done
case "$stage" in
  host | compile) export CUDA_VISIBLE_DEVICES="" ;;
  runtime) ;;
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
  CCCL_CUDA_COOP_CUTLASS_PYENV_READY=1 "$ci_dir/build_cuda_coop_cutlass.sh" -py-version "${py_version}"
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
    command -v cuobjdump >/dev/null || { echo "cuobjdump is required for final-cubin checks" >&2; exit 1; }
    nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
    for route in 0 1; do
      CUTE_DSL_USE_EXTENSION_COMPILER="$route" python -I -m pytest -v -m gpu compile/ runtime/
    done
    mapfile -t examples < <(find "$package_root/examples" -maxdepth 1 -name '*.py' ! -name '__init__.py' ! -name '_*' -print | sort)
    [[ ${#examples[@]} -gt 0 ]] || { echo "No CUTLASS examples found" >&2; exit 1; }
    for example in "${examples[@]}"; do
      python -I "$example"
    done
    ;;
  *) echo "Unhandled stage: $stage" >&2; exit 1 ;;
esac
