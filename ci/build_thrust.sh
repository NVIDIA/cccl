#!/bin/bash

# Clone Thrust repository
git clone https://github.com/NVIDIA/thrust.git

# Change to the Thrust directory
cd thrust

# Build Thrust tests
mkdir build
cd build
cmake -G ninja ..
ninja

# Run Thrust tests
#ctest --output-on-failure
