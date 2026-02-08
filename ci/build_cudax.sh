#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/build_common.sh"

print_environment_details

# If the cudax_ENABLE_CUFILE variable is specified, we don't modify it. Otherwise if we got an nvcc binary, check the
# nvcc version and if it's less than 12.9, disable the cuFile support. NVHPC Toolkit doesn't come with cuFile, too, so
# we don't enable cuFile support if the host compiler is nvc++.
if [[ -z "${cudax_ENABLE_CUFILE:-}" ]]; then
  cudax_ENABLE_CUFILE="false"
  if [[ -n "${NVCC_VERSION:-}" ]] && [[ "$(basename "${HOST_COMPILER}")" != "nvc++" ]]; then
    if util/version_compare.sh ${NVCC_VERSION} ge 12.9; then
      cudax_ENABLE_CUFILE="true"
    fi
  fi
fi

PRESET="cudax"

CMAKE_OPTIONS=(
  "-Dcudax_ENABLE_CUFILE=${cudax_ENABLE_CUFILE}"
  "-DCMAKE_CXX_STANDARD=${CXX_STANDARD}"
  "-DCMAKE_CUDA_STANDARD=${CXX_STANDARD}"
)

configure_and_build_preset "CUDA Experimental" "$PRESET" "${CMAKE_OPTIONS[*]}"

print_time_summary
