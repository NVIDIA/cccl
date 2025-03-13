#!/bin/bash

set -euo pipefail

# Ensure the script is being executed in the root cccl directory:
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/../..";

# Get the current CCCL info:
readonly cccl_repo="${PWD}"
readonly workdir="${cccl_repo}/libcudacxx/test/stdpar"

mkdir -p "${workdir}"
cd "${workdir}"

# Configure and build
rm -rf build
mkdir build
cd build
cmake -G Ninja ..
cmake --build .
ctest .
