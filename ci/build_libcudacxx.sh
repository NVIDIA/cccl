#!/bin/bash

source "$(dirname "$0")/build_common.sh"

if [ -z ${LIBCUDACXX_USE_NVRTC+x} ]; then
    LIBCUDACXX_USE_NVRTC=OFF
elif [ ${LIBCUDACXX_USE_NVRTC+x} != "ON" && ${LIBCUDACXX_USE_NVRTC+x} != "OFF"]; then
    echo "The LIBCUDACXX_USE_NVRTC environment variable must be set to either \"ON\" or \"OFF\""
    exit 1
fi

CMAKE_OPTIONS="
    -DCCCL_ENABLE_THRUST=OFF \
    -DCCCL_ENABLE_LIBCUDACXX=ON \
    -DCCCL_ENABLE_CUB=OFF \
    -DCCCL_ENABLE_TESTING=OFF \
    -DLIBCUDACXX_ENABLE_LIBCUDACXX_TESTS=ON \
    -DLIBCUDACXX_TEST_WITH_NVRTC=${LIBCUDACXX_USE_NVRTC} \
"
configure "$CMAKE_OPTIONS"

source "./sccache_stats.sh" "start"
LIBCUDACXX_SITE_CONFIG="${BUILD_DIR}/libcudacxx/test/lit.site.cfg" lit -v --no-progress-bar -Dexecutor="NoopExecutor()" -Dcompute_archs=${GPU_ARCHS} -Dstd="c++${CXX_STANDARD}" ../libcudacxx/.upstream-tests/test
source "./sccache_stats.sh" "end"
