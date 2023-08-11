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
readonly ENABLE_CUB_BENCHMARKS=${ENABLE_CUB_BENCHMARKS:=$(version_compare $NVCC_VERSION 11.5)}

if [[ $ENABLE_CUB_BENCHMARKS == "true" ]]; then
    echo "CUDA version is $NVCC_VERSION. Building CUB benchmarks."
else
    echo "CUDA version is $NVCC_VERSION. Not building CUB benchmarks because CUDA version is less than 11.5."
fi

PRESET="ci-cub-cpp$CXX_STANDARD"

# TODO Can we move the benchmark logic to CMake?
CMAKE_OPTIONS="
    -DCUB_ENABLE_BENCHMARKS="$ENABLE_CUB_BENCHMARKS"\
"

configure_and_build_preset "CUB" "$PRESET" "$CMAKE_OPTIONS"
