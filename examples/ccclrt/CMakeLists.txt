# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

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

# Default to building for the GPU on the current system
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Add all subdirectories from this directory.
file(
  GLOB subdirs
  LIST_DIRECTORIES true
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/*
)
foreach (subdir ${subdirs})
  if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
    add_subdirectory(${subdir})
  endif()
endforeach()
