#!/bin/bash

# Clone Thrust repository 
git clone --recursive --depth=1 https://github.com/NVIDIA/thrust.git

# Configure Thrust
cmake -S thrust -B thrust/build -GNinja -DTHRUST_DISABLE_ARCH_BY_DEFAULT=ON -DTHRUST_ENABLE_COMPUTE_70=ON 

# Build Thrust tests
cmake --build thrust/build


# Run Thrust tests
#ctest --output-on-failure
