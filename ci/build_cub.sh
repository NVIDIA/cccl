#!/bin/bash

# Ensure the script is being executed in its containing directory
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";

echo "Begin CUB build"
pwd

# Check if the correct number of arguments has been provided
if [ "$#" -ne 3 ]; then
    echo "Usage: ./build_cub.sh <HOST_COMPILER> <CXX_STANDARD> <GPU_ARCHS>"
    echo "The PARALLEL_LEVEL environment variable controls the amount of build parallelism. Default is the number of cores."
    echo "Example: PARALLEL_LEVEL=8 ./build_cub.sh g++-8 14 \"70\" "
    echo "Example: ./build_cub.sh clang++-8 17 \"70;75;80\" "
    exit 1
fi

# Assign command line arguments to variables
HOST_COMPILER=$1
CXX_STANDARD=$2

# Replace spaces, commas and semicolons with semicolons for CMake list
GPU_ARCHS=$(echo $3 | tr ' ,' ';')

PARALLEL_LEVEL=${PARALLEL_LEVEL:=$(nproc)}

cmake -S .. -B ../build \
    -DCCCL_ENABLE_THRUST=OFF \
    -DCCCL_ENABLE_LIBCUDACXX=OFF \
    -DCCCL_ENABLE_CUB=ON \
    -DCCCL_ENABLE_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=${HOST_COMPILER} \
    -DCMAKE_CUDA_ARCHITECTURES=${GPU_ARCHS} \
    -DCUB_ENABLE_DIALECT_CPP11=$(if [[ $CXX_STANDARD -ne 11 ]]; then echo "OFF"; else echo "ON"; fi) \
    -DCUB_ENABLE_DIALECT_CPP14=$(if [[ $CXX_STANDARD -ne 14 ]]; then echo "OFF"; else echo "ON"; fi) \
    -DCUB_ENABLE_DIALECT_CPP17=$(if [[ $CXX_STANDARD -ne 17 ]]; then echo "OFF"; else echo "ON"; fi) \
    -DCUB_ENABLE_DIALECT_CPP20=$(if [[ $CXX_STANDARD -ne 20 ]]; then echo "OFF"; else echo "ON"; fi) \
    -G Ninja

# Build the tests
cmake --build ../build 

echo "CUB build complete"
