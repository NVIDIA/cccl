#!/bin/bash

source "$(dirname "$0")/build_common.sh"

print_environment_details

PRESET="libcudacxx-cpp${CXX_STANDARD}"
CMAKE_OPTIONS=""

configure_preset libcudacxx "$PRESET" "$CMAKE_OPTIONS"

LIT_PRESET="libcudacxx-lit-cpp${CXX_STANDARD}"

source "./sccache_stats.sh" "start"
test_preset "libcudacxx (lit)" ${LIT_PRESET}
source "./sccache_stats.sh" "end"

print_time_summary
