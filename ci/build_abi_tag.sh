#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

ci_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
host_compiler=${CXX:-g++}
cxx_standard=17

while [[ $# -gt 0 ]]; do
  case "$1" in
  -cxx)
    host_compiler=${2:?Missing compiler after -cxx}
    shift 2
    ;;
  -std)
    cxx_standard=${2:?Missing standard after -std}
    shift 2
    ;;
  *)
    echo "Unknown argument: $1" >&2
    exit 1
    ;;
  esac
done

nvcc --version
"$host_compiler" --version

build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT

# Reject warnings about ignored attributes in both the CUDA and host frontends.
command=(
  nvcc
  -ccbin "$host_compiler"
  -std="c++$cxx_standard"
  -arch=sm_75
  --Werror all-warnings
  -Xcompiler=-Werror
  -c "$ci_dir/abi_tag.cu"
  -o "$build_dir/abi_tag.o"
)
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}"
