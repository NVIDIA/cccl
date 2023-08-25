#!/bin/bash

source "$(dirname "$0")/build_common.sh"

PRESET="libcudacxx-cpp${CXX_STANDARD}"
CMAKE_OPTIONS=""

configure_preset libcudacxx "$PRESET" "$CMAKE_OPTIONS"

source "./sccache_stats.sh" "start"
test_preset libcudacxx ${PRESET}
source "./sccache_stats.sh" "end"
