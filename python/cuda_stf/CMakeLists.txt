# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.30...3.31 FATAL_ERROR)

# CUDASTF ships Linux-only. Fail early on other platforms rather than producing
# an empty wheel.
if (WIN32)
  message(FATAL_ERROR "cuda-stf is only supported on Linux.")
endif()

# Must be set before project() initializes the CUDA language; otherwise CMake
# < 3.23 defaults to sm_52, which is below CCCL's minimum supported arch.
include(../../cmake/CCCLCheckCudaArchitectures.cmake)
set(
  CMAKE_CUDA_ARCHITECTURES
  "${minimum_cccl_arch}"
  CACHE STRING
  "CUDA architectures for CCCL"
)

project(cuda_stf DESCRIPTION "Python package cuda_stf" LANGUAGES CUDA CXX C)

find_package(CUDAToolkit)

set(CUDA_VERSION_MAJOR ${CUDAToolkit_VERSION_MAJOR})
set(CUDA_VERSION_DIR "cu${CUDA_VERSION_MAJOR}")
message(
  STATUS
  "Building for CUDA ${CUDA_VERSION_MAJOR}, output directory: ${CUDA_VERSION_DIR}"
)

set(_cccl_root ../..)
set(CCCL_TOPLEVEL_PROJECT ON) # Enable the developer builds

# cuda-stf only needs cccl.c.experimental.stf (and cudax). The c.parallel
# libraries belong to cuda-cccl; keep them off here.
set(CCCL_ENABLE_C_PARALLEL OFF)
set(CCCL_ENABLE_C_PARALLEL_V2 OFF)

set(CCCL_ENABLE_C_EXPERIMENTAL_STF ON)
set(CCCL_ENABLE_UNSTABLE ON)
set(CCCL_C_EXPERIMENTAL_STF_ENABLE_TESTING OFF)
set(CCCL_C_EXPERIMENTAL_STF_LIBRARY_OUTPUT_DIRECTORY ${SKBUILD_PROJECT_NAME})

# Enabling CCCL_ENABLE_UNSTABLE pulls cudax into a full developer build whose
# tests/examples/header-tests default to ON. None of those are installed into
# the wheel, so building them is wasted work. Only the cudax::cudax target
# (used by cccl.c.experimental.stf) is needed here.
set(cudax_ENABLE_TESTING OFF)
set(cudax_ENABLE_EXAMPLES OFF)
set(cudax_ENABLE_HEADER_TESTING OFF)

# Build the CCCL targets (cccl.c.experimental.stf and cudax) but do NOT ship
# the libcudacxx/CUB/Thrust headers from here: cuda-cccl already owns
# cuda/cccl/headers/include for those, and shipping them again would create
# file-ownership conflicts when both wheels are installed. We install only the
# cudax and C STF headers cuda-cccl does not provide (see below), placing them
# into the same include root so cuda.cccl.headers can discover them.
set(libcudacxx_ENABLE_INSTALL_RULES OFF)
set(CUB_ENABLE_INSTALL_RULES OFF)
set(Thrust_ENABLE_INSTALL_RULES OFF)
add_subdirectory(${_cccl_root} _parent_cccl)

# ---------------------------------------------------------------------------
# Cython toolchain
# ---------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

get_filename_component(_python_path "${Python3_EXECUTABLE}" PATH)

set(CYTHON_version_command "${Python3_EXECUTABLE}" -m cython --version)
execute_process(
  COMMAND ${CYTHON_version_command}
  OUTPUT_VARIABLE CYTHON_version_output
  ERROR_VARIABLE CYTHON_version_error
  RESULT_VARIABLE CYTHON_version_result
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_STRIP_TRAILING_WHITESPACE
)

if (NOT ${CYTHON_version_result} EQUAL 0)
  set(_error_msg "Command \"${CYTHON_version_command}\" failed with")
  set(_error_msg "${_error_msg} output:\n${CYTHON_version_error}")
  message(FATAL_ERROR "${_error_msg}")
else()
  if ("${CYTHON_version_output}" MATCHES "^[Cc]ython version ([^,]+)")
    set(CYTHON_VERSION "${CMAKE_MATCH_1}")
  else()
    if ("${CYTHON_version_error}" MATCHES "^[Cc]ython version ([^,]+)")
      set(CYTHON_VERSION "${CMAKE_MATCH_1}")
    endif()
  endif()
endif()

# -3 generates source for Python 3
# -M generates depfile
# -t cythonizes if PYX is newer than preexisting output
# -w sets working directory
set(CYTHON_FLAGS "-3 -M -t -w \"${cuda_stf_SOURCE_DIR}\"")
string(REGEX REPLACE " " ";" CYTHON_FLAGS_LIST "${CYTHON_FLAGS}")

message(STATUS "Using Cython ${CYTHON_VERSION}")

# ---------------------------------------------------------------------------
# cuda.stf._experimental (CUDASTF) extension
# ---------------------------------------------------------------------------
# Create CCCL::cudax alias for STF (normally created by cccl-config.cmake)
if (TARGET cudax::cudax AND NOT TARGET CCCL::cudax)
  add_library(CCCL::cudax ALIAS cudax::cudax)
endif()

set(_stf_install_dir "cuda/stf/_experimental/${CUDA_VERSION_DIR}")

file(MAKE_DIRECTORY "${_stf_install_dir}/cccl")

install(
  TARGETS cccl.c.experimental.stf
  DESTINATION "${_stf_install_dir}/cccl"
)

# Ship the C STF public header(s) into the same include root as the other CCCL
# headers so consumers can `#include <cccl/c/experimental/stf/stf.h>`.
install(
  DIRECTORY ${_cccl_root}/c/experimental/stf/include/cccl
  DESTINATION cuda/cccl/headers/include
  FILES_MATCHING
  PATTERN "*.h"
)

# Ship the cudax headers (e.g. cuda/experimental/places.cuh, stf.cuh) needed
# to compile C++/CUDA code against the STF C library. We install them
# explicitly rather than via cudax_ENABLE_INSTALL_RULES: that option's default
# is evaluated (in cmake/install/cudax.cmake) before CCCL_ENABLE_CUDAX is
# defined, so it cannot be reliably enabled from here.
install(
  DIRECTORY ${_cccl_root}/cudax/include/cuda
  DESTINATION cuda/cccl/headers/include
  FILES_MATCHING
  PATTERN "*.cuh"
)

set(
  stf_pyx_source_file
  "${cuda_stf_SOURCE_DIR}/cuda/stf/_experimental/_stf_bindings_impl.pyx"
)
set(
  _stf_generated_extension_src
  "${cuda_stf_BINARY_DIR}/_stf_bindings_impl.c"
)
set(_stf_depfile "${cuda_stf_BINARY_DIR}/_stf_bindings_impl.c.dep")

add_custom_command(
  OUTPUT "${_stf_generated_extension_src}"
  COMMAND "${Python3_EXECUTABLE}" -m cython
  # gersemi: off
  ARGS
    ${CYTHON_FLAGS_LIST}
    "${stf_pyx_source_file}"
    --output-file "${_stf_generated_extension_src}"
  # gersemi: on
  DEPENDS "${stf_pyx_source_file}"
  DEPFILE "${_stf_depfile}"
  COMMENT "Cythonizing ${stf_pyx_source_file} for CUDA ${CUDA_VERSION_MAJOR}"
)

set_source_files_properties(
  "${_stf_generated_extension_src}"
  PROPERTIES GENERATED TRUE
)
add_custom_target(
  cythonize_stf_bindings_impl
  ALL
  DEPENDS "${_stf_generated_extension_src}"
)

python3_add_library(
  _stf_bindings_impl
  MODULE
  WITH_SOABI
  "${_stf_generated_extension_src}"
)
add_dependencies(_stf_bindings_impl cythonize_stf_bindings_impl)
target_link_libraries(
  _stf_bindings_impl
  PRIVATE cccl.c.experimental.stf CUDA::cuda_driver
)
set_target_properties(
  _stf_bindings_impl
  PROPERTIES INSTALL_RPATH "$ORIGIN/cccl"
)

install(TARGETS _stf_bindings_impl DESTINATION "${_stf_install_dir}")
