cmake_minimum_required(VERSION 3.21)

project(
    cuda_parallel
#    VERSION ${SKBUILD_PROJECT_VERSION}
    DESCRIPTION "Python package cuda_parallel"
    LANGUAGES CUDA CXX C
)

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

find_package(Python REQUIRED COMPONENTS Development.Module)

set(CYTHON_FLAGS "-t -w \"${CMAKE_SOURCE_DIR}\"")
message(STATUS "$ENV{Cython_DIR}")
message(STATUS "$ENV{Cython_ROOT}")

set(_saved_path ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/resources/cmake)
find_package(Cython REQUIRED)
set(CMAKE_MODULE_PATH ${_saved_path})

find_package(CUDAToolkit)

add_cython_target(_cy_bindings cuda/parallel/experimental/_cy_bindings.pyx C OUTPUT_VAR _generated_c_src)
Python_add_library(_cy_bindings MODULE WITH_SOABI ${_generated_c_src})
target_link_libraries(_cy_bindings PRIVATE cccl.c.parallel CUDA::cuda_driver)
set_target_properties(_cy_bindings PROPERTIES INSTALL_RPATH "$ORIGIN/cccl")
install(TARGETS _cy_bindings DESTINATION cuda/parallel/experimental)
