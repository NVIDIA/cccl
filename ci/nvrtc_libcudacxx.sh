#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/build_common.sh"

print_environment_details


PRESET="libcudacxx-nvrtc"
CMAKE_OPTIONS="-DCMAKE_CXX_STANDARD=${CXX_STANDARD} -DCMAKE_CUDA_STANDARD=${CXX_STANDARD}"

configure_and_build_preset "libcudacxx NVRTC" "$PRESET" "$CMAKE_OPTIONS"

sccache -z > /dev/null || :
test_preset "libcudacxx NVRTC" "${PRESET}"
sccache --show-adv-stats || :
