# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21...3.31 FATAL_ERROR)

project(
  cuda_cccl_experimental
  DESCRIPTION "Python package cuda_cccl_experimental (CUDASTF)"
  LANGUAGES CUDA CXX C
)

find_package(CUDAToolkit)

set(CUDA_VERSION_MAJOR ${CUDAToolkit_VERSION_MAJOR})
set(CUDA_VERSION_DIR "cu${CUDA_VERSION_MAJOR}")
message(
  STATUS
  "Building for CUDA ${CUDA_VERSION_MAJOR}, output directory: ${CUDA_VERSION_DIR}"
)

set(_cccl_root ../..)

set(CCCL_TOPLEVEL_PROJECT ON)
set(CCCL_ENABLE_C_PARALLEL OFF)
set(CCCL_ENABLE_C_EXPERIMENTAL_STF ON)
set(CCCL_ENABLE_UNSTABLE ON)

set(CCCL_ENABLE_TESTING OFF)
set(CCCL_ENABLE_EXAMPLES OFF)
set(CCCL_ENABLE_BENCHMARKS OFF)
set(CCCL_C_EXPERIMENTAL_STF_ENABLE_TESTING OFF)
set(cudax_ENABLE_TESTING OFF)
set(cudax_ENABLE_EXAMPLES OFF)
set(cudax_ENABLE_HEADER_TESTING OFF)
set(cudax_ENABLE_CUFILE OFF)
set(CCCL_C_EXPERIMENTAL_STF_LIBRARY_OUTPUT_DIRECTORY ${SKBUILD_PROJECT_NAME})

set(libcudacxx_ENABLE_INSTALL_RULES OFF)
set(CUB_ENABLE_INSTALL_RULES OFF)
set(Thrust_ENABLE_INSTALL_RULES OFF)
add_subdirectory(${_cccl_root} _parent_cccl)

# Create CCCL::cudax alias for STF (normally created by cccl-config.cmake)
if (TARGET cudax::cudax AND NOT TARGET CCCL::cudax)
  add_library(CCCL::cudax ALIAS cudax::cudax)
endif()

# Ensure the destination directory exists
file(MAKE_DIRECTORY "cuda/stf/${CUDA_VERSION_DIR}/cccl")

# Install STF library
install(
  TARGETS cccl.c.experimental.stf
  DESTINATION cuda/stf/${CUDA_VERSION_DIR}/cccl
)

# Build and install Cython extension
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

set(CYTHON_FLAGS "-3 -M -t -w \"${cuda_cccl_experimental_SOURCE_DIR}\"")
string(REGEX REPLACE " " ";" CYTHON_FLAGS_LIST "${CYTHON_FLAGS}")

message(STATUS "STF Using Cython ${CYTHON_VERSION}")
set(
  stf_pyx_source_file
  "${cuda_cccl_experimental_SOURCE_DIR}/cuda/stf/_stf_bindings_impl.pyx"
)
set(
  _stf_generated_extension_src
  "${cuda_cccl_experimental_BINARY_DIR}/_stf_bindings_impl.c"
)
set(
  _stf_depfile
  "${cuda_cccl_experimental_BINARY_DIR}/_stf_bindings_impl.c.dep"
)

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

install(TARGETS _stf_bindings_impl DESTINATION cuda/stf/${CUDA_VERSION_DIR})
