#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/build_common.sh"

print_environment_details

PRESET="libcudacxx"
CMAKE_OPTIONS="-DCMAKE_CXX_STANDARD=${CXX_STANDARD} -DCMAKE_CUDA_STANDARD=${CXX_STANDARD}"

configure_and_build_preset libcudacxx "$PRESET" "$CMAKE_OPTIONS"

print_time_summary
