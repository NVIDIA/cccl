#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail
ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$ci_dir/.." && pwd)"
# shellcheck source=ci/util/python/common_arg_parser.sh
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"
require_py_version "Usage: $0 -py-version VERSION -stage host|frontend" || exit 1
stage=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -stage) stage="${2:?Missing stage}"; shift 2 ;;
    -stage=*) stage="${1#*=}"; shift ;;
    *) shift ;;
  esac
done
case "$stage" in
  host|frontend) ;;
  *) echo "Error: -stage must be host or frontend" >&2; exit 1 ;;
esac

# shellcheck source=ci/pyenv_helper.sh
source "$ci_dir/pyenv_helper.sh"
if [[ "${CUDA_COOP_TILE_PYENV_READY:-0}" != "1" ]]; then
  setup_python_env "$py_version" ".cccl-coop-tile-${stage}-venv"
fi
python -m pip install --quiet -r "$ci_dir/cuda_coop_tile_requirements.txt"
export PYTHONPATH="$repo_root/python/cuda_coop_tile"
export CUDA_VISIBLE_DEVICES=""
cd "$repo_root/python/cuda_coop_tile"
python "$repo_root/tile/tools/generate_tile_declarations.py" --check

if [[ "$stage" == host ]]; then
  python -m pytest -q -m "not link" \
    tests/backends/tile_python/unit \
    tests/backends/tile_cpp/api \
    tests/backends/tile_cpp/compile \
    tests/backends/tile_python/lowering/test_lowerings.py \
    tests/providers/tile/source \
    tests/integration/packaging/test_tile_header_discovery.py
else
  # This compiles public C++ declarations to TileIR without loading a provider.
  # Linking and execution require the separately supplied provider SDK.
  nvcc="${CUDA_COOP_TILE_CPP_TEST_NVCC:-${CUDACXX:-nvcc}}"
  nvcc_version="$("$nvcc" --version)"
  [[ "$nvcc_version" == *"release 13.5,"* ]] || {
    echo "Error: Tile frontend qualification requires CUDA 13.5" >&2
    exit 1
  }
  output="$repo_root/build/tile-frontend"
  mkdir -p "$output"
  printf '#include <cuda/coop/tile.cuh>\n' > "$output/compiler_mode.cu"
  for mode in disabled missing-preview; do
    flags=()
    if [[ "$mode" == missing-preview ]]; then
      flags+=(--enable-tile)
    fi
    if "$nvcc" "${flags[@]}" -std=c++20 -c \
      -I "$repo_root/tile/cpp/include" -I "$repo_root/libcudacxx/include" \
      "$output/compiler_mode.cu" -o "$output/compiler_mode.o" \
      > "$output/$mode.log" 2>&1; then
      echo "Error: Tile headers accepted a missing compiler mode: $mode" >&2
      exit 1
    fi
    grep -q 'requires nvcc --enable-tile --tile-preview=simt' "$output/$mode.log"
  done
  for name in api_sm120 scan_reduce scan_prefix reduce_axes radix_sort_sm120 radix_rank_sm120 \
    frontend_abi_sm120 exchange_smoke histogram_smoke histogram_256_smoke \
    discontinuity_flags_smoke adjacent_shuffle_sm120 merge_sort merge_sort_keys_smoke \
    run_length_decode_sm120 topk_sm120 topk_keys_smoke; do
    "$nvcc" --enable-tile --tile-preview=simt --tilebc -std=c++20 -arch=sm_120 \
      -I "$repo_root/tile/cpp/include" -I "$repo_root/libcudacxx/include" \
      "$repo_root/tile/cpp/test/support/tile_cpp_${name}.cu" -o "$output/$name.tilebc"
  done
fi
