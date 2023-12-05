#!/bin/bash

source "$(dirname "$0")/build_common.sh"

print_environment_details


PRESET="libcudacxx-nvrtc-cpp${CXX_STANDARD}"
CMAKE_OPTIONS=""

configure_and_build_preset "libcudacxx NVRTC" "$PRESET" "$CMAKE_OPTIONS"

source "./sccache_stats.sh" "start"
test_preset "libcudacxx NVRTC" "${PRESET}"
source "./sccache_stats.sh" "end"
