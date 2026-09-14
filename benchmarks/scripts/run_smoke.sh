#!/usr/bin/env bash

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -eu

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
preset="${CCCL_SMOKE_PRESET:-cub-benchmark}"
build_dir="$repo_root/build/${CCCL_BUILD_INFIX:+$CCCL_BUILD_INFIX/}$preset"

cmake -S "$repo_root" --preset "$preset"

cd "$build_dir"
export PYTHONPATH="$repo_root/benchmarks/scripts${PYTHONPATH:+:$PYTHONPATH}"
exec "$repo_root/benchmarks/scripts/run.py" --smoke "$@"
