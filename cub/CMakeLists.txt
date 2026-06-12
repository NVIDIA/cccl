if (NOT CCCL_ENABLE_CUB)
  include(cmake/CubAddSubdir.cmake)
  return()
endif()

cmake_minimum_required(VERSION 3.21)
project(CUB LANGUAGES CXX CUDA)

option(CUB_ENABLE_HEADER_TESTING "Test that all public headers compile." ON)
option(CUB_ENABLE_TESTING "Build CUB testing suite." ON)
option(CUB_ENABLE_EXAMPLES "Build CUB examples." ON)

# Opt-in: build cub::DeviceTransform's tile-DSL path (test + benches) under `nvcc --enable-tile`.
# Defaults OFF; CI never sets it, so the tile code stays gated out except in an explicit local
# --enable-tile build. Applied per-target in test/ and benchmarks/ -- never via global CMAKE_CUDA_FLAGS,
# which would also hit the C++17 c2h helper lib where cuda_tile.h hard-errors on the dialect.
option(
  CCCL_ENABLE_TILE_TRANSFORM_DISPATCH
  "Build cub::DeviceTransform's tile path (requires nvcc --enable-tile)."
  OFF
)
if (
  CCCL_ENABLE_TILE_TRANSFORM_DISPATCH
  AND "${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVIDIA"
  AND "${CMAKE_CUDA_COMPILER_VERSION}" VERSION_LESS 13.4
)
  message(
    FATAL_ERROR
    "CCCL_ENABLE_TILE_TRANSFORM_DISPATCH requires CUDA 13.4+ (nvcc --enable-tile). "
    "Found ${CMAKE_CUDA_COMPILER_VERSION}."
  )
endif()

option(CUB_ENABLE_TUNING "Build CUB tuning suite." OFF)
if ("NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  set(CUB_ENABLE_TUNING OFF)
endif()

include(cmake/CubBuildCompilerTargets.cmake)
include(cmake/CubCudaConfig.cmake)
include(cmake/CubUtilities.cmake)

cub_build_compiler_targets()

if (CUB_ENABLE_HEADER_TESTING)
  include(cmake/CubHeaderTesting.cmake)
endif()

if (CUB_ENABLE_TESTING)
  add_subdirectory(test)
endif()

if (CUB_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

if (CCCL_ENABLE_BENCHMARKS OR CUB_ENABLE_TUNING)
  add_subdirectory(benchmarks)
endif()
