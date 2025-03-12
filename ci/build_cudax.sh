#!/bin/bash

set -euo pipefail

source "$(dirname "$0")/build_common.sh"

print_environment_details

PRESET="cudax-cpp$CXX_STANDARD"

CMAKE_OPTIONS=""

# Enable extra mathlibs if we're in an extended CUDA image:
if $CCCL_CUDA_EXTENDED; then
  echo "Image with extended CUDA libs detected, enabling STF MathLibs."
  CMAKE_OPTIONS="$CMAKE_OPTIONS -Dcudax_ENABLE_CUDASTF_MATHLIBS=ON"
fi

configure_and_build_preset "CUDA Experimental" "$PRESET" "$CMAKE_OPTIONS"

print_time_summary
