#!/bin/bash

source "$(dirname "$0")/build_common.sh"

print_environment_details

PRESET="cuda-next-cpp$CXX_STANDARD"

CMAKE_OPTIONS=""

configure_and_build_preset "CudaNext" "$PRESET" "$CMAKE_OPTIONS"

print_time_summary
