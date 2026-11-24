#===----------------------------------------------------------------------===##
#
# Part of CUDA Experimental in CUDA C++ Core Libraries,
# under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#
#===----------------------------------------------------------------------===##

#[=======================================================================[.rst:
FindNCCL
--------

Find NCCL

Imported targets
^^^^^^^^^^^^^^^^

This module defines the following :prop_tgt:`IMPORTED` target(s):

``NCCL::nccl``
  The NCCL library, if found.

Result variables
^^^^^^^^^^^^^^^^

This module will set the following variables in your project:

``NCCL_FOUND``
  True if NCCL is found.
``NCCL_INCLUDE_DIRS``
  The include directories needed to use NCCL.
``NCCL_LIBRARIES``
  The libraries needed to useNCCL.
``NCCL_VERSION_STRING``
  The version of the NCCL library found. [OPTIONAL]

#]=======================================================================]

# Prefer using a Config module if it exists for this project

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)

# Also search CUDA paths for good measure
if (CUDAToolkit_ROOT)
  list(APPEND CMAKE_PREFIX_PATH ${CUDAToolkit_ROOT})
endif()

find_package(NCCL CONFIG QUIET)
if (NCCL_FOUND)
  find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_CONFIG)
  return()
endif()

find_path(NCCL_INCLUDE_DIR NAMES nccl.h)

if (NOT NCCL_LIBRARY)
  find_library(NCCL_LIBRARY_RELEASE NAMES nccl)
  find_library(NCCL_LIBRARY_DEBUG NAMES nccld)

  include(${CMAKE_ROOT}/Modules/SelectLibraryConfigurations.cmake)
  select_library_configurations(NCCL)
  unset(NCCL_FOUND) # incorrectly set by select_library_configurations
endif()

find_package_handle_standard_args(
  NCCL
  FOUND_VAR NCCL_FOUND
  REQUIRED_VARS NCCL_LIBRARY NCCL_INCLUDE_DIR
  VERSION_VAR NCCL_VERSION
)

if (NCCL_FOUND)
  set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})

  if (NOT NCCL_LIBRARIES)
    set(NCCL_LIBRARIES ${NCCL_LIBRARY})
  endif()

  if (NOT TARGET NCCL::nccl)
    add_library(NCCL::nccl UNKNOWN IMPORTED GLOBAL)
    set_target_properties(
      NCCL::nccl
      PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIRS}"
    )

    if (NCCL_LIBRARY_RELEASE)
      set_property(
        TARGET NCCL::nccl
        APPEND
        PROPERTY IMPORTED_CONFIGURATIONS RELEASE
      )
      set_target_properties(
        NCCL::nccl
        PROPERTIES IMPORTED_LOCATION_RELEASE "${NCCL_LIBRARY_RELEASE}"
      )
    endif()

    if (NCCL_LIBRARY_DEBUG)
      set_property(
        TARGET NCCL::nccl
        APPEND
        PROPERTY IMPORTED_CONFIGURATIONS DEBUG
      )
      set_target_properties(
        NCCL::nccl
        PROPERTIES IMPORTED_LOCATION_DEBUG "${NCCL_LIBRARY_DEBUG}"
      )
    endif()

    if (NOT NCCL_LIBRARY_RELEASE AND NOT NCCL_LIBRARY_DEBUG)
      set_property(
        TARGET NCCL::nccl
        APPEND
        PROPERTY IMPORTED_LOCATION "${NCCL_LIBRARY}"
      )
    endif()
  endif()
endif()
