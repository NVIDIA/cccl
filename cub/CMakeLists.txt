if (NOT CCCL_ENABLE_CUB)
  include(cmake/CubAddSubdir.cmake)
  return()
endif()

cmake_minimum_required(VERSION 3.21)
project(CUB LANGUAGES CXX CUDA)

option(CUB_ENABLE_HEADER_TESTING "Test that all public headers compile." ON)
option(CUB_ENABLE_TESTING "Build CUB testing suite." ON)
option(CUB_ENABLE_EXAMPLES "Build CUB examples." ON)

option(
  CCCL_ENABLE_EXPERIMENTAL_TILE_TRANSFORM_DISPATCH
  "EXPERIMENTAL: build cub::DeviceTransform's tile path (requires nvcc --enable-tile). This flag is experimental and may change or be removed."
  OFF
)
if (
  CCCL_ENABLE_EXPERIMENTAL_TILE_TRANSFORM_DISPATCH
  AND "${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVIDIA"
  AND "${CMAKE_CUDA_COMPILER_VERSION}" VERSION_LESS 13.4
)
  message(
    FATAL_ERROR
    "CCCL_ENABLE_EXPERIMENTAL_TILE_TRANSFORM_DISPATCH requires CUDA 13.4+ (nvcc --enable-tile). "
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
