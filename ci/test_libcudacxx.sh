#!/bin/bash

source "$(dirname "$0")/build_common.sh"

if [ -z ${LIBCUDACXX_USE_NVRTC+x} ]; then
    LIBCUDACXX_USE_NVRTC=OFF
elif [ ${LIBCUDACXX_USE_NVRTC+x} != "ON" && ${LIBCUDACXX_USE_NVRTC+x} != "OFF"]; then
    echo "The LIBCUDACXX_USE_NVRTC environment variable must be set to either \"ON\" or \"OFF\""
    exit 1
fi

cmake -S .. -B ../build \
    -DCCCL_ENABLE_THRUST=OFF \
    -DCCCL_ENABLE_LIBCUDACXX=ON \
    -DCCCL_ENABLE_CUB=OFF \
    -DCCCL_ENABLE_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=${HOST_COMPILER} \
    -DCMAKE_CUDA_ARCHITECTURES=${GPU_ARCHS} \
    -DCMAKE_CUDA_HOST_COMPILER=${HOST_COMPILER} \
    -DLIBCUDACXX_ENABLE_LIBCUDACXX_TESTS=ON \
    -Dlibcudacxx_ENABLE_INSTALL_RULES=ON \
    -DLIBCUDACXX_TEST_WITH_NVRTC=${LIBCUDACXX_USE_NVRTC} \
    -DCUB_ENABLE_INSTALL_RULES=ON \
    -DTHRUST_ENABLE_INSTALL_RULES=ON \
    -G Ninja

LIBCUDACXX_SITE_CONFIG="../build/libcudacxx/test/lit.site.cfg" lit -v --no-progress-bar -Dcompute_archs=${GPU_ARCHS} -Dstd="c++${CXX_STANDARD}" ../libcudacxx/.upstream-tests/test

