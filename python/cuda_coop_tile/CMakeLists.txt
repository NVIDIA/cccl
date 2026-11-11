# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21...3.31 FATAL_ERROR)
project(cuda_coop_tile_python LANGUAGES NONE)

# Source archives carry native library sources and CCCL headers, but never the
# separately supplied preview SDK. Creating an archive needs no native tools.
if (SKBUILD_STATE STREQUAL "sdist")
  return()
endif()
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/_tile")
  set(_tile_source "${CMAKE_CURRENT_SOURCE_DIR}/_tile")
  set(CCCL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/_cccl")
else()
  set(_tile_source "${CMAKE_CURRENT_SOURCE_DIR}/../../tile")
  get_filename_component(
    CCCL_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../.."
    ABSOLUTE
  )
endif()
set(
  CUDA_COOP_TILE_INSTALL_INCLUDE_DIR
  "cuda/coop/tile/include"
  CACHE PATH
  ""
  FORCE
)
set(
  CUDA_COOP_TILE_INSTALL_LIBRARY_DIR
  "cuda/coop/tile/_native"
  CACHE PATH
  ""
  FORCE
)
set(CUDA_COOP_TILE_INSTALL_CMAKE_DIR "cuda/coop/tile/cmake" CACHE PATH "" FORCE)
add_subdirectory("${_tile_source}" "${CMAKE_CURRENT_BINARY_DIR}/tile")

set(_distribution_version "${SKBUILD_PROJECT_VERSION_FULL}")
if (NOT _distribution_version)
  set(_distribution_version "${SKBUILD_PROJECT_VERSION}")
endif()
if (NOT _distribution_version)
  set(_distribution_version "0.1.0.dev0")
endif()
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/install_provider_manifest.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/install_provider_manifest.cmake"
  @ONLY
)
install(SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/install_provider_manifest.cmake")
