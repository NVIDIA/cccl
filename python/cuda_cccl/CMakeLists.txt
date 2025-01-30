cmake_minimum_required(VERSION 3.21...3.31 FATAL_ERROR)

# include_guard(GLOBAL)

project(
    CCCL_HEADERS
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES C CXX
    DESCRIPTION "Headers of NVIDIA CUDA Core Compute Libraries"
)

add_subdirectory(../.. _parent_cccl)

find_package(CUB REQUIRED)
find_package(Thrust REQUIRED)
find_package(libcudacxx REQUIRED)

set(_dest_incl_dir cuda/cccl/include)

install(
    DIRECTORY ${CUB_SOURCE_DIR}/cub
    DESTINATION ${_dest_incl_dir}
)
install(
    DIRECTORY ${Thrust_SOURCE_DIR}/thrust
    DESTINATION ${_dest_incl_dir}
)
install(
    DIRECTORY ${libcudacxx_SOURCE_DIR}/include
    DESTINATION ${_dest_incl_dir}
)
