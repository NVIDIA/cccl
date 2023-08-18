#!/bin/bash

source "$(dirname "$0")/build_common.sh"

PRESET="libcudacxx-cpp${CXX_STANDARD}"
CMAKE_OPTIONS=""

configure_preset libcudacxx "$PRESET" "$CMAKE_OPTIONS"

readonly TEST_PARALLEL_LEVEL=8

source "./sccache_stats.sh" "start"
LIBCUDACXX_SITE_CONFIG="${BUILD_DIR}/${PRESET}/libcudacxx/test/lit.site.cfg" lit -v -j ${TEST_PARALLEL_LEVEL} --no-progress-bar -Dcompute_archs=${GPU_ARCHS} -Dstd="c++${CXX_STANDARD}" ../libcudacxx/.upstream-tests/test
source "./sccache_stats.sh" "end"
