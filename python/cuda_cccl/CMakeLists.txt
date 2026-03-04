# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21...3.31 FATAL_ERROR)

project(cuda_cccl DESCRIPTION "Python package cuda_cccl" LANGUAGES CUDA CXX C)

find_package(CUDAToolkit)

set(CUDA_VERSION_MAJOR ${CUDAToolkit_VERSION_MAJOR})
set(CUDA_VERSION_DIR "cu${CUDA_VERSION_MAJOR}")
message(
  STATUS
  "Building for CUDA ${CUDA_VERSION_MAJOR}, output directory: ${CUDA_VERSION_DIR}"
)

# Build cccl.c.parallel and add CCCL's install rules
set(_cccl_root ../..)

# Build and install C++ library first
set(CCCL_TOPLEVEL_PROJECT ON) # Enable the developer builds
set(CCCL_ENABLE_C_PARALLEL ON) # Build the cccl.c.parallel library
# STF C interface and Python bindings are not compatible with MSVC; disable on Windows.
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(CCCL_ENABLE_C_EXPERIMENTAL_STF OFF)
  message(STATUS "STF disabled: not compatible with MSVC")
else()
  set(CCCL_ENABLE_C_EXPERIMENTAL_STF ON)
endif()
set(CCCL_ENABLE_UNSTABLE ON) # Enable unstable features

# Disable all testing, examples, and benchmarks - we only want the libraries
set(CCCL_ENABLE_TESTING OFF)
set(CCCL_ENABLE_EXAMPLES OFF)
set(CCCL_ENABLE_BENCHMARKS OFF)
set(CCCL_C_PARALLEL_ENABLE_TESTING OFF)
set(CCCL_C_EXPERIMENTAL_STF_ENABLE_TESTING OFF)
# Note: CCCL_ENABLE_CUDAX must be ON because STF depends on it (via CCCL_ENABLE_UNSTABLE)
# But disable cudax tests, examples, and header testing
set(cudax_ENABLE_TESTING OFF)
set(cudax_ENABLE_EXAMPLES OFF)
set(cudax_ENABLE_HEADER_TESTING OFF)
set(cudax_ENABLE_CUFILE OFF)
set(CCCL_C_PARALLEL_LIBRARY_OUTPUT_DIRECTORY ${SKBUILD_PROJECT_NAME})
set(CCCL_C_EXPERIMENTAL_STF_LIBRARY_OUTPUT_DIRECTORY ${SKBUILD_PROJECT_NAME})

# Just install the rest:
set(libcudacxx_ENABLE_INSTALL_RULES ON)
set(CUB_ENABLE_INSTALL_RULES ON)
set(Thrust_ENABLE_INSTALL_RULES ON)
# Install to our output location:
include(GNUInstallDirs)
set(old_libdir "${CMAKE_INSTALL_LIBDIR}") # push
set(old_includedir "${CMAKE_INSTALL_INCLUDEDIR}") # push
set(CMAKE_INSTALL_LIBDIR "cuda/cccl/headers/lib")
set(CMAKE_INSTALL_INCLUDEDIR "cuda/cccl/headers/include")
add_subdirectory(${_cccl_root} _parent_cccl)
set(CMAKE_INSTALL_LIBDIR "${old_libdir}") # pop
set(CMAKE_INSTALL_INCLUDEDIR "${old_includedir}") # pop

# Create CCCL::cudax alias for STF (normally created by cccl-config.cmake)
if (
  CCCL_ENABLE_C_EXPERIMENTAL_STF
  AND TARGET cudax::cudax
  AND NOT TARGET CCCL::cudax
)
  add_library(CCCL::cudax ALIAS cudax::cudax)
endif()

# Ensure the destination directory exists
file(MAKE_DIRECTORY "cuda/compute/${CUDA_VERSION_DIR}/cccl")
if (CCCL_ENABLE_C_EXPERIMENTAL_STF)
  file(MAKE_DIRECTORY "cuda/stf/${CUDA_VERSION_DIR}/cccl")
  # Copy all headers needed to compile against STF into the package tree.
  # This makes cuda.stf.get_include_path() self-contained: downstream packages
  # get version-consistent headers from a single -I path.
  set(_stf_include_dest "${CMAKE_CURRENT_SOURCE_DIR}/cuda/stf/include")
  file(MAKE_DIRECTORY "${_stf_include_dest}")
  file(COPY ${_cccl_root}/c/experimental/stf/include/  DESTINATION ${_stf_include_dest})
  file(COPY ${_cccl_root}/cudax/include/               DESTINATION ${_stf_include_dest})
  file(COPY ${_cccl_root}/libcudacxx/include/           DESTINATION ${_stf_include_dest})
  file(COPY ${_cccl_root}/cub/cub/                      DESTINATION ${_stf_include_dest}/cub)
  file(COPY ${_cccl_root}/thrust/thrust/                DESTINATION ${_stf_include_dest}/thrust)
endif()

# Install version-specific binaries
if (CCCL_ENABLE_C_EXPERIMENTAL_STF)
  install(
    TARGETS cccl.c.experimental.stf
    DESTINATION cuda/stf/${CUDA_VERSION_DIR}/cccl
  )

  # Install STF C API header and cudax headers so downstream packages
  # can compile C++ extensions against the exact same ABI.
  install(
    DIRECTORY ${_stf_include_dest}/
    DESTINATION cuda/stf/include
  )

  # Generate build config so downstream packages can compile with matching flags.
  # Queried at runtime via cuda.stf.get_compile_flags().
  get_target_property(_stf_cuda_std cccl.c.experimental.stf CUDA_STANDARD)
  get_target_property(_stf_cxx_std  cccl.c.experimental.stf CXX_STANDARD)
  get_target_property(_stf_cuda_archs cccl.c.experimental.stf CUDA_ARCHITECTURES)
  if (NOT _stf_cuda_archs)
    set(_stf_cuda_archs "${CMAKE_CUDA_ARCHITECTURES}")
  endif()
  string(REPLACE ";" "," _stf_cuda_archs_csv "${_stf_cuda_archs}")
  set(_stf_build_config "{
  \"cuda_standard\": ${_stf_cuda_std},
  \"cxx_standard\": ${_stf_cxx_std},
  \"cuda_architectures\": \"${_stf_cuda_archs_csv}\",
  \"cuda_toolkit_version\": \"${CUDAToolkit_VERSION}\",
  \"defines\": [\"CCCL_C_EXPERIMENTAL\"],
  \"nvcc_flags\": [\"--expt-relaxed-constexpr\", \"--extended-lambda\"]
}
")
  set(_stf_build_config_path "${CMAKE_CURRENT_BINARY_DIR}/stf_build_config.json")
  file(WRITE "${_stf_build_config_path}" "${_stf_build_config}")
  install(
    FILES "${_stf_build_config_path}"
    DESTINATION cuda/stf/${CUDA_VERSION_DIR}
  )
  # Also copy into source tree for editable installs
  file(COPY "${_stf_build_config_path}" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/cuda/stf/${CUDA_VERSION_DIR}")
endif()

install(
  TARGETS cccl.c.parallel
  DESTINATION cuda/compute/${CUDA_VERSION_DIR}/cccl
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

# -3 generates source for Python 3
# -M generates depfile
# -t cythonizes if PYX is newer than preexisting output
# -w sets working directory
set(CYTHON_FLAGS "-3 -M -t -w \"${cuda_cccl_SOURCE_DIR}\"")
string(REGEX REPLACE " " ";" CYTHON_FLAGS_LIST "${CYTHON_FLAGS}")

message(STATUS "Using Cython ${CYTHON_VERSION}")
set(pyx_source_file "${cuda_cccl_SOURCE_DIR}/cuda/compute/_bindings_impl.pyx")

set(_generated_extension_src "${cuda_cccl_BINARY_DIR}/_bindings_impl.c")
set(_depfile "${cuda_cccl_BINARY_DIR}/_bindings_impl.c.dep")

# Custom Cython compilation command for version-specific target
add_custom_command(
  OUTPUT "${_generated_extension_src}"
  COMMAND "${Python3_EXECUTABLE}" -m cython
  # gersemi: off
  ARGS
    ${CYTHON_FLAGS_LIST}
    "${pyx_source_file}"
    --output-file "${_generated_extension_src}"
  # gersemi: on
  DEPENDS "${pyx_source_file}"
  DEPFILE "${_depfile}"
  COMMENT "Cythonizing ${pyx_source_file} for CUDA ${CUDA_VERSION_MAJOR}"
)

set_source_files_properties(
  "${_generated_extension_src}"
  PROPERTIES GENERATED TRUE
)
add_custom_target(
  cythonize_bindings_impl
  ALL
  DEPENDS "${_generated_extension_src}"
)

python3_add_library(
  _bindings_impl
  MODULE
  WITH_SOABI
  "${_generated_extension_src}"
)

add_dependencies(_bindings_impl cythonize_bindings_impl)
target_link_libraries(
  _bindings_impl
  PRIVATE #
    cccl.c.parallel
    CUDA::cuda_driver
)
set_target_properties(_bindings_impl PROPERTIES INSTALL_RPATH "$ORIGIN/cccl")

install(TARGETS _bindings_impl DESTINATION cuda/compute/${CUDA_VERSION_DIR})

if (CCCL_ENABLE_C_EXPERIMENTAL_STF)
  message(STATUS "STF Using Cython ${CYTHON_VERSION}")
  set(
    stf_pyx_source_file
    "${cuda_cccl_SOURCE_DIR}/cuda/stf/_stf_bindings_impl.pyx"
  )
  set(
    _stf_generated_extension_src
    "${cuda_cccl_BINARY_DIR}/_stf_bindings_impl.c"
  )
  set(_stf_depfile "${cuda_cccl_BINARY_DIR}/_stf_bindings_impl.c.dep")
  add_custom_command(
    OUTPUT "${_stf_generated_extension_src}"
    COMMAND "${Python3_EXECUTABLE}" -m cython
    ARGS
      ${CYTHON_FLAGS_LIST} "${stf_pyx_source_file}" --output-file
      ${_stf_generated_extension_src}
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
endif()
