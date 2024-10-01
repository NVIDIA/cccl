# 3.15 is the minimum for including the project with add_subdirectory.
# 3.21 is the minimum for the developer build.
cmake_minimum_required(VERSION 3.15)

if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_LIST_DIR}")
  set(cudax_TOPLEVEL_PROJECT ON)
endif()

# This must be done before any languages are enabled:
if (cudax_TOPLEVEL_PROJECT)
  cmake_minimum_required(VERSION 3.21)
endif()

project(cudax LANGUAGES CUDA CXX)

if (NOT cudax_TOPLEVEL_PROJECT)
  include(cmake/cudaxAddSubdir.cmake)
  return()
endif()

option(cudax_ENABLE_HEADER_TESTING "Test that CUDA Experimental's public headers compile." ON)
option(cudax_ENABLE_TESTING "Build CUDA Experimental's tests." ON)
option(cudax_ENABLE_SAMPLES "Build CUDA Experimental's samples." ON)

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

if (cudax_ENABLE_SAMPLES)
  include(ExternalProject)
  ExternalProject_Add(cudax_samples
    PREFIX samples
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/samples"
    BUILD_ALWAYS ON
    INSTALL_COMMAND cmake -E echo "Skipping install step.")
  add_dependencies(cudax.all cudax_samples)
endif()
