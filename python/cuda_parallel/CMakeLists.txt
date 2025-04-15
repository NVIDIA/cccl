cmake_minimum_required(VERSION 3.21)

project(
    cuda_parallel
#    VERSION ${SKBUILD_PROJECT_VERSION}
    DESCRIPTION "Python package cuda_parallel"
    LANGUAGES CUDA CXX C
)

find_package(CUDAToolkit)

set(_cccl_root ../..)

include(${_cccl_root}/cmake/AppendOptionIfAvailable.cmake)
include(${_cccl_root}/cmake/CCCLConfigureTarget.cmake)
include(${_cccl_root}/cmake/CCCLBuildCompilerTargets.cmake)
cccl_build_compiler_targets()

set(CCCL_ENABLE_C ON)
set(CCCL_C_PARALLEL_LIBRARY_OUTPUT_DIRECTORY ${SKBUILD_PROJECT_NAME})
add_subdirectory(${_cccl_root} _parent_cccl)

install(
    TARGETS cccl.c.parallel
    DESTINATION cuda/parallel/experimental/cccl
)

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
set(CYTHON_FLAGS "-3 -M -t -w \"${cuda_parallel_SOURCE_DIR}\"")
string(REGEX REPLACE " " ";" CYTHON_FLAGS_LIST "${CYTHON_FLAGS}")

message(STATUS "Using Cython ${CYTHON_VERSION}")
set(pyx_source_file "${cuda_parallel_SOURCE_DIR}/cuda/parallel/experimental/_bindings.pyx")
set(_generated_extension_src "${cuda_parallel_BINARY_DIR}/_bindings.c")
set(_depfile "${cuda_parallel_BINARY_DIR}/_bindings.c.dep")
add_custom_command(
    OUTPUT "${_generated_extension_src}"
    COMMAND "${Python3_EXECUTABLE}" -m cython
    ARGS ${CYTHON_FLAGS_LIST} "${pyx_source_file}" --output-file ${_generated_extension_src}
    DEPENDS "${pyx_source_file}"
    DEPFILE "${_depfile}"
)
set_source_files_properties("${_generated_extension_src}" PROPERTIES GENERATED TRUE)
add_custom_target(cythonize_bindings ALL
    DEPENDS "${_generated_extension_src}"
)

Python3_add_library(_bindings MODULE WITH_SOABI "${_generated_extension_src}")
add_dependencies(_bindings cythonize_bindings)
target_link_libraries(_bindings PRIVATE cccl.c.parallel CUDA::cuda_driver)
set_target_properties(_bindings PROPERTIES INSTALL_RPATH "$ORIGIN/cccl")

install(TARGETS _bindings DESTINATION cuda/parallel/experimental)
