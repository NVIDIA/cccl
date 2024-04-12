# 3.15 is the minimum for including the project with add_subdirectory.
# 3.21 is the minimum for the developer build.
cmake_minimum_required(VERSION 3.15)

# FIXME Tests currently fail when this is defined to Release. Temporary WAR to validate CMake changes:
set(CMAKE_BUILD_TYPE "")

if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_LIST_DIR}")
  set(CudaNext_TOPLEVEL_PROJECT ON)
endif()

# This must be done before any languages are enabled:
if (CudaNext_TOPLEVEL_PROJECT)
  cmake_minimum_required(VERSION 3.21)
endif()

project(CudaNext LANGUAGES CUDA)

option(CudaNext_ENABLE_INSTALL_RULES "Enable installation of CudaNext" ${CudaNext_TOPLEVEL_PROJECT})
if (CudaNext_ENABLE_INSTALL_RULES)
  include(cmake/CudaNextInstallRules.cmake)
endif()

if (NOT CudaNext_TOPLEVEL_PROJECT)
  include(cmake/CudaNextAddSubdir.cmake)
  return()
endif()

option(CudaNext_ENABLE_HEADER_TESTING "Test that all public headers compile." ON)
option(CudaNext_ENABLE_TESTING "Build CUDA Next tests." ON)

include(cmake/CudaNextBuildCompilerTargets.cmake)
include(cmake/CudaNextBuildTargetList.cmake)

CudaNext_build_compiler_targets()
CudaNext_build_target_list()

if (CudaNext_ENABLE_HEADER_TESTING)
  include(cmake/CudaNextHeaderTesting.cmake)
endif()

if (CudaNext_ENABLE_TESTING)
  include(CTest)
  enable_testing() # Must be in root directory
  add_subdirectory(test)
endif()
