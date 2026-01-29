# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21)

project(
  cuda_compute_cpp_benchmarks
  DESCRIPTION "Compare cuda.compute with C++ benchmarks"
  LANGUAGES CXX CUDA
)

# Configure paths to CCCL root
set(CCCL_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../../")
get_filename_component(CCCL_SOURCE_DIR "${CCCL_SOURCE_DIR}" ABSOLUTE)
set(CCCL_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/_cccl_build")

message(STATUS "CCCL Source: ${CCCL_SOURCE_DIR}")
message(STATUS "CCCL Binary: ${CCCL_BINARY_DIR}")

# Add CCCL CMake modules to path
list(APPEND CMAKE_MODULE_PATH "${CCCL_SOURCE_DIR}/cmake")

# Include CCCL dependency helpers and build utilities
include(AppendOptionIfAvailable)
include("${CCCL_SOURCE_DIR}/cmake/CCCLGetDependencies.cmake")
include("${CCCL_SOURCE_DIR}/cmake/CCCLConfigureTarget.cmake")
include("${CCCL_SOURCE_DIR}/cmake/CCCLBuildCompilerTargets.cmake")

# Get dependencies using CCCL's functions
cccl_get_nvbench()
cccl_get_nvbench_helper()
cccl_get_cub()
cccl_get_thrust()
cccl_get_libcudacxx()
cccl_get_cudatoolkit()

# Create compiler interface target
cccl_build_compiler_targets()

# GPU architectures
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(STATUS "CMAKE_CUDA_ARCHITECTURES not set, using 'native'")
  set(CMAKE_CUDA_ARCHITECTURES native)
else()
  message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Set build type to Release if not specified
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ============================================================================
# CUB benchmarks from: cub/benchmarks/bench
# ============================================================================

function(add_cub_benchmark target_name source_file)
  add_executable(${target_name} ${source_file})

  # Compile with TUNE_BASE=1 for baseline (no tuning)
  target_compile_definitions(${target_name} PRIVATE TUNE_BASE=1)

  # Link all dependencies
  target_link_libraries(
    ${target_name}
    PRIVATE
      CUB::CUB
      Thrust::Thrust
      libcudacxx::libcudacxx
      nvbench::main
      cccl.nvbench_helper
  )

  # Extended lambda support (required by CUB benchmarks)
  target_compile_options(${target_name} PRIVATE "--extended-lambda")

  # C++/CUDA standards (matching CCCL requirements)
  set_target_properties(
    ${target_name}
    PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
  )

  # Add CUB benchmark directory to includes for common.h
  get_filename_component(bench_dir "${source_file}" DIRECTORY)
  target_include_directories(${target_name} PRIVATE "${bench_dir}")

  message(STATUS "Added CUB benchmark target: ${target_name}")
endfunction()

# Transform benchmarks
add_cub_benchmark(nvbench_fill_cpp
  "${CCCL_SOURCE_DIR}/cub/benchmarks/bench/transform/fill.cu"
)
