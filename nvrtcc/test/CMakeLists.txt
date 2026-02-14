##===----------------------------------------------------------------------===##
##
## Part of nvrtcc in CUDA Core Compute Libraries,
## under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
##
##===----------------------------------------------------------------------===##

set(CMAKE_BUILD_TYPE Debug)

cmake_minimum_required(VERSION 3.21)

project(nvrtc-testing LANGUAGES CXX CUDA)

add_custom_target(nvrtcc.test)
enable_testing()

function(add_exec_test name)
  add_executable(${name} ${ARGN})
  add_dependencies(nvrtcc.test ${name})
  target_compile_options(${name} PRIVATE -use-nvrtc)
endfunction()

function(add_compile_only_test name)
  add_library(${name} OBJECT ${ARGN})
  add_dependencies(nvrtcc.test ${name})
  target_compile_options(${name} PRIVATE -use-nvrtc)
endfunction()

add_subdirectory(nvcc_options)
add_subdirectory(nvrtcc_options)
add_subdirectory(predefined_macros)
