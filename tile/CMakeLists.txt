# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21)
project(cuda_coop_tile LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if (NOT CCCL_SOURCE_DIR)
  get_filename_component(
    CCCL_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    ABSOLUTE
  )
endif()

# The declarations can be consumed before enabling the native provider.
add_library(cuda_coop_tile INTERFACE)
add_library(cuda-coop-tile::tile ALIAS cuda_coop_tile)
target_compile_features(cuda_coop_tile INTERFACE cuda_std_20)
target_compile_options(
  cuda_coop_tile
  INTERFACE "$<$<COMPILE_LANGUAGE:CUDA>:--enable-tile;--tile-preview=simt>"
)
if (WIN32)
  target_compile_options(
    cuda_coop_tile
    INTERFACE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>"
  )
endif()
target_include_directories(
  cuda_coop_tile
  INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include"
    "${CCCL_SOURCE_DIR}/libcudacxx/include"
)
