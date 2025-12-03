# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This demo provides an example of how to configure a project to use Thrust while selecting
# the device system as a configuration option. The device system is selected by setting the
# CMake option `CCCL_THRUST_DEVICE_SYSTEM={CUDA, OMP, TBB, CPP}` for CUDA, OpenMP, Intel Threading
# Building Blocks (TBB), and serial C++, respectively. If no option is provided, the default is `CUDA`.
#
# See the accompanying README.md for more information and build instructions.

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(ThrustFlexibleDeviceSystemDemo CXX)

# This example uses the CMake Package Manager (CPM) to simplify fetching CCCL from GitHub
# For more information, see https://github.com/cpm-cmake/CPM.cmake
include(cmake/CPM.cmake)

# We define these as variables so they can be overridden in CI to pull from a PR instead of CCCL `main`
# In your project, these variables are unnecessary and you can just use the values directly
set(
  CCCL_REPOSITORY
  "https://github.com/NVIDIA/cccl"
  CACHE STRING
  "GitHub repository to fetch CCCL from"
)
set(CCCL_TAG "main" CACHE STRING "Git tag/branch to fetch from CCCL repository")

# This will automatically clone CCCL from GitHub and make the exported cmake targets available.
# The default `CCCL::Thrust` target will be configured to use the system device defined by
# `CCCL_THRUST_DEVICE_SYSTEM`.
CPMAddPackage(NAME CCCL GIT_REPOSITORY "${CCCL_REPOSITORY}" GIT_TAG ${CCCL_TAG})

# CUDA specific setup
if (CCCL_THRUST_DEVICE_SYSTEM STREQUAL "CUDA")
  # Need to explicitly enable the CUDA language for the project.
  # Note that the project(...) command earlier only enables CXX by default.
  enable_language(CUDA)

  # Optional: Call find_package(CCCL) again after enabling new languages
  # to update compatibility flags for newly detected compilers.
  find_package(CCCL CONFIG REQUIRED)

  # Compile for the native CUDA arch if not specified:
  if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
endif()

# Creates a cmake executable target for the main program
add_executable(example_program example.cpp)

# Thrust requires at least C++17:
target_compile_features(example_program PUBLIC cxx_std_17)
if (CCCL_THRUST_DEVICE_SYSTEM STREQUAL "CUDA")
  target_compile_features(example_program PUBLIC cuda_std_17)
endif()

# By default, CMake inspects the source file extension to determine whether to use C++ or CUDA
# compilers. We can override this behavior by using source file properties. Here, we tell CMake
# to compile this C++ (.cpp) file with the CUDA compiler when using the CUDA device system:
if (CCCL_THRUST_DEVICE_SYSTEM STREQUAL "CUDA")
  set_source_files_properties(example.cpp PROPERTIES LANGUAGE CUDA)
endif()

# "Links" the CCCL Cmake target to the `example_program` executable. This configures everything needed to use
# CCCL headers, including setting up include paths, compiler flags, Thrust host/device configuration, etc.
target_link_libraries(example_program PRIVATE CCCL::CCCL)

# This is only relevant for internal testing and not needed by end users.
include(CTest)
enable_testing()
add_test(NAME example_program COMMAND example_program)
set_tests_properties(
  example_program
  PROPERTIES
    PASS_REGULAR_EXPRESSION
      "Detected device system: ${CCCL_THRUST_DEVICE_SYSTEM}"
)
