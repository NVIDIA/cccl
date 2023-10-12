#!/bin/bash

source "$(dirname "$0")/build_common.sh"

# CUB benchmarks require at least CUDA nvcc 11.5 for int128
# Returns "true" if the first version is greater than or equal to the second
version_compare() {
    if [[ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" == "$2" ]]; then
        echo "true"
    else
        echo "false"
    fi
}

ENABLE_CUB_BENCHMARKS="false"
ENABLE_CUB_RDC="false"
if [[ "$CUDA_COMPILER" == *nvcc* ]]; then
    ENABLE_CUB_RDC="true"
    NVCC_VERSION=$($CUDA_COMPILER --version | grep release | awk '{print $6}' | cut -c2-)
    if [[ $(version_compare $NVCC_VERSION 11.5) == "true" ]]; then
        ENABLE_CUB_BENCHMARKS="true"
        echo "nvcc version is $NVCC_VERSION. Building CUB benchmarks."
    else
        echo "nvcc version is $NVCC_VERSION. Not building CUB benchmarks because nvcc version is less than 11.5."
    fi
else
    echo "nvcc version is not determined (likely using a non-NVCC compiler). Not building CUB benchmarks."
fi

CMAKE_OPTIONS="
    -DCCCL_ENABLE_THRUST=OFF \
    -DCCCL_ENABLE_LIBCUDACXX=OFF \
    -DCCCL_ENABLE_CUB=ON \
    -DCCCL_ENABLE_TESTING=OFF \
    -DCUB_ENABLE_DIALECT_CPP11=$(if [[ $CXX_STANDARD -ne 11 ]]; then echo "OFF"; else echo "ON"; fi) \
    -DCUB_ENABLE_DIALECT_CPP14=$(if [[ $CXX_STANDARD -ne 14 ]]; then echo "OFF"; else echo "ON"; fi) \
    -DCUB_ENABLE_DIALECT_CPP17=$(if [[ $CXX_STANDARD -ne 17 ]]; then echo "OFF"; else echo "ON"; fi) \
    -DCUB_ENABLE_DIALECT_CPP20=$(if [[ $CXX_STANDARD -ne 20 ]]; then echo "OFF"; else echo "ON"; fi) \
    -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT=ON \
    -DCUB_IGNORE_DEPRECATED_CPP_DIALECT=ON \
    -DCUB_ENABLE_BENCHMARKS="$ENABLE_CUB_BENCHMARKS"\
    -DCUB_ENABLE_RDC_TESTS="$ENABLE_CUB_RDC" \
"

configure_and_build "CUB" "$CMAKE_OPTIONS"
