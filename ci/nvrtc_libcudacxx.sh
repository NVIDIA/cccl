#!/bin/bash

source "$(dirname "$0")/build_common.sh"

CMAKE_OPTIONS="
    -DCCCL_ENABLE_THRUST=OFF \
    -DCCCL_ENABLE_LIBCUDACXX=ON \
    -DCCCL_ENABLE_CUB=OFF \
    -DCCCL_ENABLE_TESTING=OFF \
    -DLIBCUDACXX_ENABLE_LIBCUDACXX_TESTS=ON \
    -DLIBCUDACXX_TEST_WITH_NVRTC=ON \
"
configure "$CMAKE_OPTIONS"

readonly TEST_PARALLEL_LEVEL=8

source "./sccache_stats.sh" "start"
LIBCUDACXX_SITE_CONFIG="${BUILD_DIR}/libcudacxx/test/lit.site.cfg" lit -v -j ${TEST_PARALLEL_LEVEL} --no-progress-bar -Dcompute_archs=${GPU_ARCHS} -Dstd="c++${CXX_STANDARD}" ../libcudacxx/.upstream-tests/test
source "./sccache_stats.sh" "end"
