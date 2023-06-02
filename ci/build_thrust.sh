#!/bin/bash

# Ensure the script is being executed in its containing directory
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";


# Check if the correct number of arguments has been provided
if [ "$#" -ne 3 ]; then
    echo "Usage: ./build_thrust.sh <HOST_COMPILER> <CXX_STANDARD> <GPU_ARCHS>"
    echo "The PARALLEL_LEVEL environment variable controls the amount of build parallelism. Default is the number of cores."
    echo "Example: PARALLEL_LEVEL=8 ./build_thrust.sh g++-8 14 \"70\" "
    echo "Example: ./build_thrust.sh clang++-8 17 \"70;75;80\" "
    exit 1
fi

# Assign command line arguments to variables
HOST_COMPILER=$(which $1)
CXX_STANDARD=$2

# Replace spaces, commas and semicolons with semicolons for CMake list
GPU_ARCHS=$(echo $3 | tr ' ,' ';')

PARALLEL_LEVEL=${PARALLEL_LEVEL:=$(nproc)}

echo "========================================"
echo "Begin Thrust build"
echo "pwd=$(pwd)"
echo "HOST_COMPILER=$HOST_COMPILER"
echo "CXX_STANDARD=$CXX_STANDARD"
echo "GPU_ARCHS=$GPU_ARCHS"
echo "PARALLEL_LEVEL=$PARALLEL_LEVEL"
echo "========================================"

cmake -S .. -B ../build \
      -DCCCL_ENABLE_THRUST=ON \
      -DCCCL_ENABLE_LIBCUDACXX=OFF \
      -DCCCL_ENABLE_CUB=OFF \
      -DCCCL_ENABLE_TESTING=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=${HOST_COMPILER} \
      -DCMAKE_CUDA_ARCHITECTURES=${GPU_ARCHS} \
      -DTHRUST_ENABLE_MULTICONFIG=ON \
      -DTHRUST_MULTICONFIG_ENABLE_DIALECT_CPP11=$(if [[ $CXX_STANDARD -ne 11 ]]; then echo "OFF"; else echo "ON"; fi) \
      -DTHRUST_MULTICONFIG_ENABLE_DIALECT_CPP14=$(if [[ $CXX_STANDARD -ne 14 ]]; then echo "OFF"; else echo "ON"; fi) \
      -DTHRUST_MULTICONFIG_ENABLE_DIALECT_CPP17=$(if [[ $CXX_STANDARD -ne 17 ]]; then echo "OFF"; else echo "ON"; fi) \
      -DTHRUST_MULTICONFIG_ENABLE_DIALECT_CPP20=$(if [[ $CXX_STANDARD -ne 20 ]]; then echo "OFF"; else echo "ON"; fi) \
      -DTHRUST_MULTICONFIG_WORKLOAD=SMALL \
      -G Ninja
      # TODO: Add this back after Thrust removes the check against it.
      #-DCMAKE_CUDA_HOST_COMPILER=${HOST_COMPILER} \

# Build the tests
cmake --build ../build --parallel ${PARALLEL_LEVEL} 

echo "Thrust build complete"
