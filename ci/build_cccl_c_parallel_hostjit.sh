#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/build_common.sh"

print_environment_details

# libnvfatbin is required by hostjit but is not included in the base rapidsai devcontainer
# image. Detect the installed CTK version and install the matching package if missing.
if [[ "$(uname -s)" == "Linux" ]] && ! ldconfig -p 2>/dev/null | grep -q libnvfatbin; then
  CTK_DEB_VER=$(nvcc --version 2>/dev/null \
    | grep -oP 'release \K[0-9]+\.[0-9]+' | tr '.' '-')
  if [[ -n "$CTK_DEB_VER" ]]; then
    echo "Installing libnvfatbin-dev-${CTK_DEB_VER}..."
    sudo apt-get update -y
    sudo apt-get install -y --no-install-recommends "libnvfatbin-dev-${CTK_DEB_VER}"
  else
    echo "WARNING: could not determine CTK version; skipping libnvfatbin install"
  fi
fi

PRESET="cccl-c-parallel-hostjit"

CMAKE_OPTIONS=()
if test -n "${CXX_STANDARD:+x}"; then
    CMAKE_OPTIONS+=("-DCMAKE_CXX_STANDARD=${CXX_STANDARD}" "-DCMAKE_CUDA_STANDARD=${CXX_STANDARD}")
fi

configure_and_build_preset "CCCL C Parallel Library (HostJIT)" "$PRESET" "${CMAKE_OPTIONS[@]}"

print_time_summary
