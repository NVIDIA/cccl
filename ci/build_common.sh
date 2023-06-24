#!/bin/bash

set -euo pipefail

# Ensure the script is being executed in its containing directory
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";

# Check if the correct number of arguments has been provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <HOST_COMPILER> <CXX_STANDARD> <GPU_ARCHS>"
    echo "The PARALLEL_LEVEL environment variable controls the amount of build parallelism. Default is the number of cores."
    echo "Example: PARALLEL_LEVEL=8 $0 g++-8 14 \"70\" "
    echo "Example: $0 clang++-8 17 \"70;75;80-virtual\" "
    exit 1
fi

# Assign command line arguments to variables
readonly HOST_COMPILER=$(which $1)
readonly CXX_STANDARD=$2

# Replace spaces, commas and semicolons with semicolons for CMake list
readonly GPU_ARCHS=$(echo $3 | tr ' ,' ';')

readonly PARALLEL_LEVEL=${PARALLEL_LEVEL:=$(nproc)}

BUILD_DIR=${BUILD_DIR:-/tmp/build/cccl}

COMMON_CMAKE_OPTIONS="
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=${CXX_STANDARD} \
    -DCMAKE_CUDA_STANDARD=${CXX_STANDARD} \
    -DCMAKE_CXX_COMPILER=${HOST_COMPILER} \
    -DCMAKE_CUDA_HOST_COMPILER=${HOST_COMPILER} \
    -DCMAKE_CUDA_ARCHITECTURES=${GPU_ARCHS} \
"

echo "========================================"
echo "Begin build"
echo "pwd=$(pwd)"
echo "HOST_COMPILER=$HOST_COMPILER"
echo "CXX_STANDARD=$CXX_STANDARD"
echo "GPU_ARCHS=$GPU_ARCHS"
echo "PARALLEL_LEVEL=$PARALLEL_LEVEL"
echo "BUILD_DIR=$BUILD_DIR"
echo "========================================"

function configure(){
    local CMAKE_OPTIONS=$1
    cmake -S .. -B $BUILD_DIR $COMMON_CMAKE_OPTIONS $CMAKE_OPTIONS -G Ninja
}

function build(){
    local BUILD_NAME=$1
    source "./sccache_stats.sh" start
    cmake --build $BUILD_DIR --parallel $PARALLEL_LEVEL
    echo "${BUILD_NAME} build complete"
    source "./sccache_stats.sh" end
}

function configure_and_build() {
    local BUILD_NAME=$1
    local CMAKE_OPTIONS=$2
    configure "$CMAKE_OPTIONS"
    build "$BUILD_NAME"
}
