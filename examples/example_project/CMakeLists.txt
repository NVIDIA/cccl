cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(CCCLDemo CUDA)

# This example uses the CMake Package Manager (CPM) to simplify fetching CCCL from GitHub
# For more information, see https://github.com/cpm-cmake/CPM.cmake
include(cmake/CPM.cmake)

# This will automatically clone CCCL from GitHub and make the exported cmake targets available
CPMAddPackage(
    NAME CCCL
    GITHUB_REPOSITORY nvidia/cccl
    GIT_TAG main
)

# Default to building for the GPU on the current system
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(example_project example.cu)
target_link_libraries(example_project PRIVATE CCCL::CCCL)

# Used by CCCL CI for regression testing:
include(CTest)
enable_testing()
add_test(NAME example_project COMMAND example_project)
