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

string(REPLACE ";" "\\\;" arches_escaped "${CMAKE_CUDA_ARCHITECTURES}")

set(
  cmake_opts
  "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
  "-DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
  "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
  "-DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
  "-DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER}"
  "-DCMAKE_CUDA_ARCHITECTURES=${arches_escaped}"
)

set(
  CCCL_EXAMPLE_CPM_REPOSITORY
  "${CCCL_SOURCE_DIR}"
  CACHE STRING
  "Git repository used for CPM examples."
)
set(
  CCCL_EXAMPLE_CPM_TAG
  "HEAD"
  CACHE STRING
  "Git tag/branch used for CPM examples."
)

set(
  cmake_cpm_opts
  "-DCCCL_REPOSITORY=${CCCL_EXAMPLE_CPM_REPOSITORY}"
  "-DCCCL_TAG=${CCCL_EXAMPLE_CPM_TAG}"
)

if (MSVC)
  message(STATUS "Skipping example build-and-test CTests on MSVC")
  return()
endif()

cccl_add_compile_test(
  test_name
  cccl.example
  basic
  "default"
  ${cmake_opts}
  ${cmake_cpm_opts}
)

find_package(CUDAToolkit REQUIRED)

if (CUDAToolkit_VERSION_MAJOR VERSION_GREATER_EQUAL 12)
  cccl_add_compile_test(
    test_name
    cccl.example
    cudax
    "default"
    ${cmake_opts}
    ${cmake_cpm_opts}
  )

  cccl_add_compile_test(
    test_name
    cccl.example
    cudax_stf
    "default"
    ${cmake_opts}
    ${cmake_cpm_opts}
  )
endif() # CTK > 12.0

foreach (DEVICE_SYSTEM IN ITEMS CUDA OMP TBB CPP)
  cccl_add_compile_test(
    test_name
    cccl.example
    thrust_flexible_device_system
    "${DEVICE_SYSTEM}"
    "-DCCCL_THRUST_DEVICE_SYSTEM=${DEVICE_SYSTEM}"
    ${cmake_opts}
    ${cmake_cpm_opts}
  )
endforeach()
