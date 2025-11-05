#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/build_common.sh"

print_environment_details


PRESET="libcudacxx-nvrtc-cpp${CXX_STANDARD}"
CMAKE_OPTIONS=""

configure_and_build_preset "libcudacxx NVRTC" "$PRESET" "$CMAKE_OPTIONS"

sccache -z > /dev/null || :
test_preset "libcudacxx NVRTC" "${PRESET}"
sccache --show-adv-stats || :
