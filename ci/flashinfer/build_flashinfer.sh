#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../.."
readonly cccl_repo="${PWD}"
readonly workdir="${cccl_repo}/build/${CCCL_BUILD_INFIX:-}/flashinfer"

git config --global --add safe.directory "${cccl_repo}"
mkdir -p "${workdir}"
echo "CCCL SHA: $(git rev-parse HEAD)"

cd "${workdir}"
echo "::group::Preparing FlashInfer sources..."
rm -rf flashinfer
git clone --depth 1 https://github.com/flashinfer-ai/flashinfer.git flashinfer
cd flashinfer
export FLASHINFER_WORKSPACE_BASE="${PWD}"
echo "FlashInfer SHA: $(git rev-parse HEAD)"
git submodule update --init --depth 1 -- 3rdparty/cutlass 3rdparty/spdlog 3rdparty/cccl

# FlashInfer's JIT adds these vendored headers with -I ahead of the CUDA Toolkit.
git -C 3rdparty/cccl fetch --depth 1 "${cccl_repo}" HEAD
git -C 3rdparty/cccl checkout --detach FETCH_HEAD
echo "::endgroup::"

echo "::group::Installing FlashInfer build dependencies..."
python --version
# shellcheck source=/dev/null
source scripts/setup_test_env.sh
# shellcheck source=/dev/null
source scripts/jit_cache_build_common.sh
compute_jit_cache_parallelism
BUILD_NVEP=0 install_flashinfer_editable
python -m pip freeze
echo "::endgroup::"

export FLASHINFER_CUDA_ARCH_LIST="9.0a 10.0a 10.7a 12.0f"

echo "::group::Verifying FlashInfer CCCL headers..."
python - <<'PY'
from pathlib import Path

from flashinfer.jit.cpp_ext import get_cccl_includes

root = Path("3rdparty/cccl")
expected = [(root / path).resolve(strict=True) for path in ("cub", "libcudacxx/include", "thrust")]
actual = [Path(path).resolve(strict=True) for path in get_cccl_includes()]
if actual != expected:
    raise SystemExit(f"Unexpected CCCL headers: {actual}; expected {expected}")
print(f"Verified FlashInfer CCCL include paths: {actual}")
PY

# shellcheck disable=SC2154 # Set by compute_jit_cache_parallelism.
echo "MAX_JOBS=${MAX_JOBS}, FLASHINFER_NVCC_THREADS=${FLASHINFER_NVCC_THREADS}"
nvcc --version
echo "::endgroup::"

echo "::group::Building FlashInfer AOT kernels..."
time python -u -m flashinfer.aot
echo "::endgroup::"
