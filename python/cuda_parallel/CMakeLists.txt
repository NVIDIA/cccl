cmake_minimum_required(VERSION 3.21)

project(
    cuda_parallel
#    VERSION ${SKBUILD_PROJECT_VERSION}
    DESCRIPTION "Python package cuda_parallel"
    LANGUAGES CUDA CXX
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
