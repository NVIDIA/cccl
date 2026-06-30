#===----------------------------------------------------------------------===//
#
# Part of libcu++, the C++ Standard Library for your entire system,
# under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#
#===----------------------------------------------------------------------===//

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Default to building for the GPU on the current system.
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(IMAGE_PIPELINE_EXAMPLE CUDA CXX)

# This example requires CUDA 13.1+ for the libcu++ runtime APIs.
find_package(CUDAToolkit REQUIRED)
if (CUDAToolkit_VERSION VERSION_LESS 13.1)
  message(
    STATUS
    "Skipping image_pipeline example: requires CUDA 13.1+ (found ${CUDAToolkit_VERSION})"
  )
  return()
endif()

# This example uses the CMake Package Manager (CPM) to simplify fetching CCCL from GitHub
# For more information, see https://github.com/cpm-cmake/CPM.cmake
include(cmake/CPM.cmake)

# We define these as variables so they can be overridden in CI to pull from a PR instead of CCCL `main`
# In your project, these variables are unnecessary and you can just use the values directly
set(
  CCCL_REPOSITORY
  "https://github.com/NVIDIA/cccl"
  CACHE STRING
  "Git repository to fetch CCCL from"
)
set(CCCL_TAG "main" CACHE STRING "Git tag/branch to fetch from CCCL repository")

# This will automatically clone CCCL from GitHub and make the exported cmake targets available
CPMAddPackage(
  NAME CCCL
  GIT_REPOSITORY "${CCCL_REPOSITORY}"
  GIT_TAG ${CCCL_TAG}
  GIT_SHALLOW ON
)

# Image processing pipeline — a multi-file example showcasing the libcu++
# runtime APIs: device selection, memory pools, buffers, copy_bytes,
# fill_bytes, double-buffered streams, events, timed events, and CUB
# operations (histogram, transform, transform-reduce, block reduce).
add_executable(image_pipeline main.cu detail.cu)
target_include_directories(image_pipeline PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(image_pipeline PRIVATE cuda_std_17)
target_link_libraries(image_pipeline PRIVATE CCCL::CCCL)
# Device lambdas (used in CUB transform/reduce ops) require extended lambda support.
target_compile_options(
  image_pipeline
  PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>
)

option(
  IMAGE_PIPELINE_ENABLE_RUNTIME_TEST
  "Enable the image_pipeline CTest runtime test. Requires a GPU and several GB of host/device memory."
  OFF
)

if (IMAGE_PIPELINE_ENABLE_RUNTIME_TEST)
  include(CTest)
  enable_testing()
  add_test(NAME image_pipeline COMMAND image_pipeline)
endif()
