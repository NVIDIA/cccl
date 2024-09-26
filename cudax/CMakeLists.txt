# 3.15 is the minimum for including the project with add_subdirectory.
# 3.21 is the minimum for the developer build.
cmake_minimum_required(VERSION 3.15)

# This must be done before any languages are enabled:
if (CCCL_ENABLE_CUDAX)
  cmake_minimum_required(VERSION 3.21)
endif()

project(cudax LANGUAGES CUDA CXX)

if (NOT CCCL_ENABLE_CUDAX)
  include(cmake/cudaxAddSubdir.cmake)
  return()
endif()

option(cudax_ENABLE_HEADER_TESTING "Test that CUDA Experimental's public headers compile." ON)
option(cudax_ENABLE_TESTING "Build CUDA Experimental's tests." ON)

include(cmake/cudaxBuildCompilerTargets.cmake)
include(cmake/cudaxBuildTargetList.cmake)

cudax_build_compiler_targets()
cudax_build_target_list()

if (cudax_ENABLE_HEADER_TESTING)
  include(cmake/cudaxHeaderTesting.cmake)
endif()

if (cudax_ENABLE_TESTING)
  include(CTest)
  enable_testing() # Must be in root directory
  add_subdirectory(test)
endif()
