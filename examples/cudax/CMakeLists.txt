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

cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(CUDAX_SAMPLES CUDA CXX)

# This example uses the CMake Package Manager (CPM) to simplify fetching CCCL from GitHub
# For more information, see https://github.com/cpm-cmake/CPM.cmake
include(cmake/CPM.cmake)

# We define these as variables so they can be overridden in CI to pull from a PR instead of CCCL `main`
# In your project, these variables are unnecessary and you can just use the values directly
set(CCCL_REPOSITORY "https://github.com/NVIDIA/cccl" CACHE STRING "Git repository to fetch CCCL from")
set(CCCL_TAG "main" CACHE STRING "Git tag/branch to fetch from CCCL repository")

# This will automatically clone CCCL from GitHub and make the exported cmake targets available
CPMAddPackage(
  NAME CCCL
  GIT_REPOSITORY "${CCCL_REPOSITORY}"
  GIT_TAG ${CCCL_TAG}
  GIT_SHALLOW ON
  # The following is required to make the `CCCL::cudax` target available:
  OPTIONS "CCCL_ENABLE_UNSTABLE ON"
)

# Default to building for the GPU on the current system
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_library(cudax_samples_interface INTERFACE)

target_compile_definitions(
  cudax_samples_interface INTERFACE
  LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
)

target_link_libraries(cudax_samples_interface INTERFACE CCCL::CCCL CCCL::cudax)

if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  # mdspan on windows only works in C++20 mode
  target_compile_features(cudax_samples_interface INTERFACE cxx_std_20)

  # Require the conforming preprocessor
  target_compile_options(cudax_samples_interface INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus /Zc:preprocessor>
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=/Zc:__cplusplus -Xcompiler=/Zc:preprocessor>
  )

  if (MSVC_TOOLSET_VERSION LESS 143)
    # winbase.h(9572): warning C5105: macro expansion producing 'defined' has undefined behavior
    target_compile_options(cudax_samples_interface INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:/wd5105>
      $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=/wd5105>
    )
  endif()
endif()

# The vector_add sample demonstrates a simple CUDA kernel that adds two vectors
add_executable(vector_add vector_add/vector_add.cu)
target_link_libraries(vector_add PUBLIC cudax_samples_interface)

# This is only relevant for internal testing and not needed by end users.
include(CTest)
enable_testing()
add_test(NAME vector_add COMMAND vector_add)
