# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(CUDAX_SAMPLES CUDA CXX)

# This example uses the CMake Package Manager (CPM) to simplify fetching CCCL from GitHub
# For more information, see https://github.com/cpm-cmake/CPM.cmake
include(cmake/CPM.cmake)

# We define these as variables so they can be overriden in CI to pull from a PR instead of CCCL `main`
# In your project, these variables are unncessary and you can just use the values directly
set(CCCL_REPOSITORY "nvidia/cccl" CACHE STRING "GitHub repository to fetch CCCL from")
set(CCCL_TAG "main" CACHE STRING "Git tag/branch to fetch from CCCL repository")

# This will automatically clone CCCL from GitHub and make the exported cmake targets available
CPMAddPackage(
  NAME CCCL
  GITHUB_REPOSITORY ${CCCL_REPOSITORY}
  GIT_TAG ${CCCL_TAG}
  GIT_SHALLOW ON
  OPTIONS "CCCL_ENABLE_UNSTABLE ON"
)

# Default to building for the GPU on the current system
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# Creates a cmake executable target for the main program
add_executable(vector_add vector_add/vector_add.cu)

# "Links" the CCCL::cudax CMake target to the `vector_add` executable. This
# configures everything needed to use CCCL's headers, including setting up
# include paths, compiler flags, etc.
target_link_libraries(vector_add
  PUBLIC
    CCCL::cudax
    CCCL::CCCL
    CCCL::Thrust
    CCCL::libcudacxx
  INTERFACE cudax.compiler_interface
)

# TODO: These are temporary until the main branch catches up with the latest changes
target_compile_definitions(vector_add PUBLIC LIBCUDACXX_ENABLE_EXCEPTIONS)

if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  # mdspan on windows only works in C++20 mode
  target_compile_features(vector_add PUBLIC cxx_std_20)

  # cudax requires dim3 to be usable from a constexpr context, and the CUDART headers require
  # __cplusplus to be defined for this to work:
  target_compile_options(vector_add PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus /Zc:preprocessor>
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=/Zc:__cplusplus -Xcompiler=/Zc:preprocessor>
  )
endif()

# This is only relevant for internal testing and not needed by end users.
include(CTest)
enable_testing()
add_test(NAME vector_add COMMAND vector_add)
