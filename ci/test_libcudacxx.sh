#!/bin/bash

echo "build libcudacxx"

# Require a compiler to be specified
#if [ $# -ne 1 ]; then
#    echo "Usage: $0 <CXX_COMPILER>"
#    exit 1
#fi

# First argument is the CXX compiler
#CXX_COMPILER=$1

# Clone Thrust repository 
#git clone --recursive --depth=1 https://github.com/NVIDIA/thrust.git

# Configure Thrust
#cmake -S thrust -B thrust/build -GNinja -DTHRUST_DISABLE_ARCH_BY_DEFAULT=ON -DTHRUST_ENABLE_COMPUTE_70=ON #-DCMAKE_CXX_COMPILER=${CXX_COMPILER}


# Build Thrust tests
#cmake --build thrust/build


# Run Thrust tests
#ctest --output-on-failure
