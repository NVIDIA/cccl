#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/build_common.sh"

print_environment_details

BUILD_NAME="clang-tidy"
PRESET="all-tidy"
CMAKE_OPTIONS=("-DCMAKE_CXX_STANDARD=${CXX_STANDARD}" "-DCMAKE_CUDA_STANDARD=${CXX_STANDARD}")
# Clang does not understand -G, passed by all-dev-debug which all-tidy derives from
CMAKE_OPTIONS+=("-DCMAKE_CUDA_FLAGS=")
# TODO(jfaibussowit)
#
# STF seems to trip clang-cuda up pretty heavily. It's unclear whether this is because STF
# hasn't been compiled against clang-cuda before or whether it's an issue with clang-cuda
# itself.
CMAKE_OPTIONS+=("-Dcudax_ENABLE_CUDASTF=OFF")
CMAKE_OPTIONS+=("-Dcudax_ENABLE_PLACES=OFF")

# todo(dabayer): Re-enable OpenMP thrust builds for clang-tidy.
CMAKE_OPTIONS+=("-DTHRUST_MULTICONFIG_ENABLE_SYSTEM_OMP=OFF")

# Cannot use configure_and_build_preset because that does not allow us to pass additional
# arguments to the build command.
configure_preset "${BUILD_NAME}" "${PRESET}" "${CMAKE_OPTIONS[@]}"
# Keep going after errors, we want CI to unearth all clang-tidy errors in one go
BUILD_OPTIONS=(-- -k 0)
build_preset "${BUILD_NAME}" "${PRESET}" "${BUILD_OPTIONS[@]}"

print_time_summary
