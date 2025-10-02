#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/build_common.sh"

print_environment_details

ENABLE_CCCL_BENCHMARKS="false"
ENABLE_CUB_RDC="false"

if [[ "$CUDA_COMPILER" == *nvcc* ]]; then
    ENABLE_CUB_RDC="true"
    NVCC_VERSION=$($CUDA_COMPILER --version | grep release | awk '{print $6}' | cut -c2-)
    if [[ -n "${DISABLE_CUB_BENCHMARKS}" ]]; then
        echo "Benchmarks have been forcefully disabled."
    else
        ENABLE_CCCL_BENCHMARKS="true"
        echo "nvcc version is $NVCC_VERSION. Building CUB benchmarks."
    fi
else
    echo "Not building with NVCC, disabling RDC and benchmarks."
fi

# Building for all arches will exceed available memory on CI systems.
# Reduce build parallelism:
# if [[ "${CUDA_ARCHS:-}" == "all" ]]; then
#     if [[ -n "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]]; then
#         if [[ "$CMAKE_BUILD_PARALLEL_LEVEL" -gt 1 ]]; then
#             CMAKE_BUILD_PARALLEL_LEVEL=$((CMAKE_BUILD_PARALLEL_LEVEL / 2))
#             echo "Reducing CMAKE_BUILD_PARALLEL_LEVEL to $CMAKE_BUILD_PARALLEL_LEVEL for CUB build with '-arch all'."
#             export CMAKE_BUILD_PARALLEL_LEVEL
#         fi
#     fi
# fi

if [[ "$HOST_COMPILER" == *icpc* || "$HOST_COMPILER" == *nvhpc* ]]; then
    ENABLE_CCCL_BENCHMARKS="false"
fi

PRESET="cub-cpp$CXX_STANDARD"

CMAKE_OPTIONS="
    -DCCCL_ENABLE_BENCHMARKS="$ENABLE_CCCL_BENCHMARKS"\
    -DCUB_ENABLE_RDC_TESTS="$ENABLE_CUB_RDC" \
"

configure_and_build_preset "CUB" "$PRESET" "$CMAKE_OPTIONS"

# Create test artifacts:
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    run_command "📦  Packaging test artifacts" /home/coder/cccl/ci/upload_cub_test_artifacts.sh
fi

print_time_summary
