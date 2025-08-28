# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21...3.31 FATAL_ERROR)

project(
    cuda_cccl
    DESCRIPTION "Python package cuda_cccl"
    LANGUAGES CUDA CXX C
)

find_package(CUDAToolkit)

set(CUDA_VERSION_MAJOR ${CUDAToolkit_VERSION_MAJOR})
set(CUDA_VERSION_DIR "cu${CUDA_VERSION_MAJOR}")
message(STATUS "Building for CUDA ${CUDA_VERSION_MAJOR}, output directory: ${CUDA_VERSION_DIR}")

set(_cccl_root ../..)

include(${_cccl_root}/cmake/AppendOptionIfAvailable.cmake)
include(${_cccl_root}/cmake/CCCLConfigureTarget.cmake)
include(${_cccl_root}/cmake/CCCLBuildCompilerTargets.cmake)
include(${_cccl_root}/cmake/CCCLGetDependencies.cmake)
cccl_build_compiler_targets()

# Build and install C++ library first
set(CCCL_ENABLE_C ON)
set(CCCL_C_PARALLEL_LIBRARY_OUTPUT_DIRECTORY ${SKBUILD_PROJECT_NAME})
add_subdirectory(${_cccl_root} _parent_cccl)

# Now we can find CUB and other components
find_package(CUB REQUIRED)
find_package(Thrust REQUIRED)
find_package(libcudacxx REQUIRED)

# Install headers
set(_dest_incl_dir cuda/cccl/headers/include)

# No end slash: create ${_dest_inc_dir}/cub
install(
    DIRECTORY ${CUB_SOURCE_DIR}/cub
    DESTINATION ${_dest_incl_dir}
)
# No end slash: create ${_dest_inc_dir}/thrust
install(
    DIRECTORY ${Thrust_SOURCE_DIR}/thrust
    DESTINATION ${_dest_incl_dir}
)
# Slash at the end: copy content of
#                   include/ into ${_dest_inc_dir}/
install(
    DIRECTORY ${libcudacxx_SOURCE_DIR}/include/
    DESTINATION ${_dest_incl_dir}
)

# ensure the destination directory exists
file(MAKE_DIRECTORY "cuda/cccl/parallel/experimental/${CUDA_VERSION_DIR}/cccl")

# Install version-specific binaries
install(
    TARGETS cccl.c.parallel
    DESTINATION cuda/cccl/parallel/experimental/${CUDA_VERSION_DIR}/cccl
)

# Build and install Cython extension
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

get_filename_component(_python_path "${Python3_EXECUTABLE}" PATH)

set(CYTHON_version_command "${Python3_EXECUTABLE}" -m cython --version)
execute_process(COMMAND ${CYTHON_version_command}
    OUTPUT_VARIABLE CYTHON_version_output
    ERROR_VARIABLE CYTHON_version_error
    RESULT_VARIABLE CYTHON_version_result
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)

if(NOT ${CYTHON_version_result} EQUAL 0)
    set(_error_msg "Command \"${CYTHON_version_command}\" failed with")
    set(_error_msg "${_error_msg} output:\n${CYTHON_version_error}")
    message(FATAL_ERROR "${_error_msg}")
else()
    if("${CYTHON_version_output}" MATCHES "^[Cc]ython version ([^,]+)")
        set(CYTHON_VERSION "${CMAKE_MATCH_1}")
    else()
        if("${CYTHON_version_error}" MATCHES "^[Cc]ython version ([^,]+)")
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
set(pyx_source_file "${cuda_cccl_SOURCE_DIR}/cuda/cccl/parallel/experimental/_bindings_impl.pyx")

set(_generated_extension_src "${cuda_cccl_BINARY_DIR}/_bindings_impl.c")
set(_depfile "${cuda_cccl_BINARY_DIR}/_bindings_impl.c.dep")

# Custom Cython compilation command for version-specific target
add_custom_command(
    OUTPUT "${_generated_extension_src}"
    COMMAND "${Python3_EXECUTABLE}" -m cython
    ARGS ${CYTHON_FLAGS_LIST} "${pyx_source_file}" --output-file "${_generated_extension_src}"
    DEPENDS "${pyx_source_file}"
    DEPFILE "${_depfile}"
    COMMENT "Cythonizing ${pyx_source_file} for CUDA ${CUDA_VERSION_MAJOR}"
)

set_source_files_properties("${_generated_extension_src}" PROPERTIES GENERATED TRUE)
add_custom_target(cythonize_bindings_impl ALL
    DEPENDS "${_generated_extension_src}"
)

Python3_add_library(_bindings_impl MODULE WITH_SOABI "${_generated_extension_src}")
add_dependencies(_bindings_impl cythonize_bindings_impl)
target_link_libraries(_bindings_impl PRIVATE cccl.c.parallel CUDA::cuda_driver)
set_target_properties(_bindings_impl PROPERTIES INSTALL_RPATH "$ORIGIN/cccl")

install(TARGETS _bindings_impl DESTINATION cuda/cccl/parallel/experimental/${CUDA_VERSION_DIR})
