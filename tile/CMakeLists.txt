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

set(
  CUDA_COOP_TILE_SDK_ROOT
  ""
  CACHE PATH
  "Separately supplied Tile linkage-provider SDK"
)
set(
  CUDA_COOP_TILE_INTERFACE_TRANSLATOR
  ""
  CACHE FILEPATH
  "CUDA Tile 13.5 declaration translator"
)
set(
  CUDA_COOP_TILE_INSTALL_INCLUDE_DIR
  "${CMAKE_INSTALL_INCLUDEDIR}"
  CACHE PATH
  "Tile and CCCL header installation directory"
)
set(
  CUDA_COOP_TILE_INSTALL_LIBRARY_DIR
  "${CMAKE_INSTALL_LIBDIR}/cuda-coop-tile"
  CACHE PATH
  "TileLib installation directory"
)
set(
  CUDA_COOP_TILE_INSTALL_CMAKE_DIR
  "${CMAKE_INSTALL_LIBDIR}/cmake/cuda-coop-tile"
  CACHE PATH
  "Tile CMake package installation directory"
)
option(CUDA_COOP_TILE_BUILD_TESTS "Build Tile C++ runtime tests" OFF)

foreach (_input CUDA_COOP_TILE_SDK_ROOT CUDA_COOP_TILE_INTERFACE_TRANSLATOR)
  if (NOT ${_input} AND DEFINED ENV{${_input}})
    set(${_input} "$ENV{${_input}}")
  endif()
  if (NOT ${_input})
    message(
      FATAL_ERROR
      "${_input} must identify the separately supplied CUDA 13.5 provider build input"
    )
  endif()
endforeach()

find_package(CUDAToolkit 13.5 REQUIRED)
if (NOT CUDAToolkit_VERSION VERSION_LESS 13.6)
  message(
    FATAL_ERROR
    "This TileLib targets the CUDA 13.5 SIMT preview; use a CUDA 13.5 toolkit"
  )
endif()
find_program(
  CUDA_COOP_TILE_TILEIRAS
  tileiras
  HINTS "${CUDAToolkit_BIN_DIR}"
  NO_DEFAULT_PATH
  REQUIRED
)

foreach (
  _file
  include/tile_ir/linkage_provider/LinkageProviderAPI.h
  include/tile_ir/linkage_provider/TileIRBytecodeBuilder.h
  include/linkage_providers/TileLinkageProvider.h
  include/linkage_providers/ImplCatalogBuilder.h
  src/TileIRBytecodeBuilder.c
)
  if (NOT EXISTS "${CUDA_COOP_TILE_SDK_ROOT}/${_file}")
    message(FATAL_ERROR "The supplied Tile provider SDK is missing ${_file}")
  endif()
endforeach()
if (NOT EXISTS "${CUDA_COOP_TILE_INTERFACE_TRANSLATOR}")
  message(
    FATAL_ERROR
    "CUDA_COOP_TILE_INTERFACE_TRANSLATOR does not name a file"
  )
endif()

if (CUDA_COOP_TILE_BUILD_TESTS)
  enable_testing()
endif()
add_subdirectory(cpp)

if (NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(
    CUDA_COOP_TILE_INTERFACE_FILE
    "${CUDA_COOP_TILE_INTERFACE_FILE}"
    PARENT_SCOPE
  )
  set(CUDA_COOP_TILE_LIBRARY_FILE "${CUDA_COOP_TILE_LIBRARY_FILE}" PARENT_SCOPE)
endif()
