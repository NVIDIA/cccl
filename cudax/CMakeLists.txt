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

project(cudax LANGUAGES CUDA)

option(cudax_ENABLE_INSTALL_RULES "Enable installation of CUDA Experimental." ${cudax_TOPLEVEL_PROJECT})
if (cudax_ENABLE_INSTALL_RULES)
  include(cmake/cudaxInstallRules.cmake)
endif()

if (NOT cudax_TOPLEVEL_PROJECT)
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
