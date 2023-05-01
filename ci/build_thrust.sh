#!/bin/bash

# Clone Thrust repository
git clone https://github.com/NVIDIA/thrust.git

# Change to the Thrust directory
cd thrust

# Build Thrust tests
mkdir build
cd build
cmake -GNinja -DTHRUST_DISABLE_ARCH_BY_DEFAULT=ON -DTHRUST_ENABLE_COMPUTE_70=ON ..
ninja

# Run Thrust tests
#ctest --output-on-failure
