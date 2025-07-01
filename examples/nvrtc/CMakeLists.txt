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

project(CCCLDemo CUDA CXX)

find_package(CUDAToolkit REQUIRED)
# This example uses the CMake Package Manager (CPM) to simplify fetching CCCL from GitHub
# For more information, see https://github.com/cpm-cmake/CPM.cmake
include(cmake/CPM.cmake)

# We define these as variables so they can be overridden in CI to pull from a PR instead of CCCL `main`
# In your project, these variables are unnecessary and you can just use the values directly
set(CCCL_REPOSITORY "nvidia/cccl" CACHE STRING "GitHub repository to fetch CCCL from")
set(CCCL_TAG "main" CACHE STRING "Git tag/branch to fetch from CCCL repository")

# This will automatically clone CCCL from GitHub and make the exported cmake targets available
CPMAddPackage(
    NAME CCCL
    GITHUB_REPOSITORY ${CCCL_REPOSITORY}
    GIT_TAG ${CCCL_TAG}
)

# Default to building for the GPU on the current system
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Creates a cmake executable target for the main program
add_executable(nvrtc_example nvrtc.cpp)

# "Links" the CCCL Cmake target to the example executable. This configures everything needed to use
# CCCL headers, including setting up include paths, compiler flags, etc.
target_link_libraries(nvrtc_example PRIVATE CCCL::CCCL)
target_link_libraries(nvrtc_example PRIVATE CUDA::cuda_driver CUDA::nvrtc)

# Set the C++ dialect
target_compile_features(nvrtc_example PRIVATE cxx_std_20)


# Get include paths for NVRTC and pass them to the translation unit
set(includes "")
get_target_property(libcudacxx_include CCCL::libcudacxx INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(cub_include CCCL::CUB INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(thrust_include Thrust::Thrust INTERFACE_INCLUDE_DIRECTORIES)
message("NVCC_INCLUDES: ${libcudacxx_include} ${cub_include} ${thrust_include}")
set(includes "${includes}\"-I${libcudacxx_include}\",")
set(includes "${includes}\"-I${cub_include}\",")
set(includes "${includes}\"-I${thrust_include}\",") # TODO: Make Thrust NVRTC safe


# Tell NVRTC which architectures and include locations are required
# Uncomment to compile for SASS
# target_compile_definitions(nvrtc_example PRIVATE "EXTERNAL_NVRTC_ARCH=sm_80")
target_compile_definitions(nvrtc_example PRIVATE "EXTERNAL_NVRTC_ARCH=compute_80")
target_compile_definitions(nvrtc_example PRIVATE "EXTERNAL_NVRTC_ARGS=${includes}")

# This is only relevant for internal testing and not needed by end users.
include(CTest)
enable_testing()
add_test(NAME nvrtc_example COMMAND nvrtc_example)
