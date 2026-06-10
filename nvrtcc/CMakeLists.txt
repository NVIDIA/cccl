##===----------------------------------------------------------------------===##
##
## Part of nvrtcc in CUDA Core Compute Libraries,
## under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
##
##===----------------------------------------------------------------------===##

cmake_minimum_required(VERSION 3.21)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

project(nvrtcc LANGUAGES CXX)

# Set output directory.
set(nvrtcc_exec_output_dir ${nvrtcc_BINARY_DIR}/bin)

find_package(CUDAToolkit REQUIRED)

# Find libcufilt.
find_library(
  nvrtcc_cufilt_lib
  "cufilt"
  PATHS "${CUDAToolkit_LIBRARY_DIR}"
  NO_DEFAULT_PATH
)
if (NOT nvrtcc_cufilt_lib)
  message(FATAL_ERROR "nvrtcc: cu++filt library (libcufilt.a) not found.")
endif()

# Add nvrtcc_device_compiler target.
add_executable(nvrtcc_device_compiler src/nvrtcc_device_compiler.cpp)
target_compile_features(nvrtcc_device_compiler PRIVATE cxx_std_17)
target_include_directories(
  nvrtcc_device_compiler
  PRIVATE "${CUDAToolkit_INCLUDE_DIRS}"
)
target_link_libraries(nvrtcc_device_compiler PRIVATE ${nvrtcc_cufilt_lib})
set_target_properties(
  nvrtcc_device_compiler
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${nvrtcc_exec_output_dir}"
)

# Add nvrtcc target that copies nvrtcc_cicc to the output directory and depends on nvrtcc_device_compiler.
add_custom_target(
  nvrtcc
  ALL
  COMMAND
    ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/bin
    ${nvrtcc_exec_output_dir}
)
add_dependencies(nvrtcc nvrtcc_device_compiler)
