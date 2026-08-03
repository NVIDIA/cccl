# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21...4.3 FATAL_ERROR)

project(cccl_headers DESCRIPTION "Python package cccl-headers" LANGUAGES NONE)

set(CCCL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(CCCL_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(CCCL_TOPLEVEL_PROJECT OFF)
set(CCCL_ENABLE_CUDAX ON)

# Install all header families and their relocatable CMake packages without
# configuring any CCCL compiler targets.
set(CCCL_ENABLE_INSTALL_RULES ON)
set(libcudacxx_ENABLE_INSTALL_RULES ON)
set(CUB_ENABLE_INSTALL_RULES ON)
set(Thrust_ENABLE_INSTALL_RULES ON)
set(cudax_ENABLE_INSTALL_RULES ON)

# Defining these before CCCLInstallRules loads GNUInstallDirs avoids probing a
# target architecture in this deliberately language-free project.
set(CMAKE_INSTALL_INCLUDEDIR "cuda/cccl/headers/include" CACHE PATH "" FORCE)
set(CMAKE_INSTALL_LIBDIR "cuda/cccl/headers/lib" CACHE PATH "" FORCE)
include("${CCCL_SOURCE_DIR}/cmake/CCCLInstallRules.cmake")

# CCCLInstallRules currently selects CUDAX's public ``*.cuh`` headers. CUDAX
# also has public ``*.h`` implementation headers referenced by those entry
# points, so install that non-overlapping suffix explicitly.
install(
  DIRECTORY "${CCCL_SOURCE_DIR}/cudax/include/cuda"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING
  PATTERN "*.h"
)

install(
  FILES "${CCCL_SOURCE_DIR}/LICENSE"
  DESTINATION cuda/cccl/headers/licenses
)
install(
  FILES "${CCCL_SOURCE_DIR}/cub/LICENSE.TXT"
  DESTINATION cuda/cccl/headers/licenses/cub
)
install(
  FILES "${CCCL_SOURCE_DIR}/cudax/LICENSE.TXT"
  DESTINATION cuda/cccl/headers/licenses/cudax
)
install(
  FILES "${CCCL_SOURCE_DIR}/libcudacxx/LICENSE.TXT"
  DESTINATION cuda/cccl/headers/licenses/libcudacxx
)
install(
  FILES "${CCCL_SOURCE_DIR}/thrust/LICENSE"
  DESTINATION cuda/cccl/headers/licenses/thrust
)
