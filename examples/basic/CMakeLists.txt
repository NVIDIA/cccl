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

project(CCCLDemo CUDA)

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
)

# Default to building for the GPU on the current system
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Creates a cmake executable target for the main program
add_executable(example_project example.cu)

# "Links" the CCCL Cmake target to the `example_project` executable. This configures everything needed to use
# CCCL headers, including setting up include paths, compiler flags, etc.
target_link_libraries(example_project PRIVATE CCCL::CCCL)

# This is only relevant for internal testing and not needed by end users.
include(CTest)
enable_testing()
add_test(NAME example_project COMMAND example_project)
