# 3.15 is the minimum for including the project with add_subdirectory.
# 3.21 is the minimum for the developer build.
cmake_minimum_required(VERSION 3.15)

# This must be done before any languages are enabled:
if (CCCL_ENABLE_THRUST)
  cmake_minimum_required(VERSION 3.21)
endif()

project(Thrust LANGUAGES NONE)

# This must appear after our Compiler Hacks or else CMake will delete the cache
# and reconfigure from scratch.
# This must also appear before the installation rules, as it is required by the
# GNUInstallDirs CMake module.
enable_language(CXX)

# Support adding Thrust to a parent project via add_subdirectory.
# See examples/cmake/add_subdir/CMakeLists.txt for details.
if (NOT CCCL_ENABLE_THRUST)
  include(cmake/ThrustAddSubdir.cmake)
  return()
endif()

# We use 3.17 features when building our tests, etc.
cmake_minimum_required(VERSION 3.17)

option(THRUST_ENABLE_HEADER_TESTING "Test that all public headers compile." "ON")
option(THRUST_ENABLE_TESTING "Build Thrust testing suite." "ON")
option(THRUST_ENABLE_EXAMPLES "Build Thrust examples." "ON")
option(THRUST_ENABLE_BENCHMARKS "Build Thrust runtime benchmarks." "${CCCL_ENABLE_BENCHMARKS}")

# Allow the user to optionally select offset type dispatch to fixed 32 or 64 bit types
set(THRUST_DISPATCH_TYPE "Dynamic" CACHE STRING "Select Thrust offset type dispatch." FORCE)
set_property(CACHE THRUST_DISPATCH_TYPE PROPERTY STRINGS "Dynamic" "Force32bit" "Force64bit")

# Check if we're actually building anything before continuing. If not, no need
# to search for deps, etc. This is a common approach for packagers that just
# need the install rules. See GH issue NVIDIA/thrust#1211.
if (NOT (THRUST_ENABLE_HEADER_TESTING OR
         THRUST_ENABLE_TESTING OR
         THRUST_ENABLE_EXAMPLES OR
         THRUST_ENABLE_BENCHMARKS))
  return()
endif()

#include first:
include(cmake/ThrustUtilities.cmake)

include(cmake/ThrustBuildCompilerTargets.cmake)
include(cmake/ThrustBuildTargetList.cmake)
include(cmake/ThrustFindThrust.cmake)
include(cmake/ThrustMultiConfig.cmake)

# Add cache string options for CMAKE_BUILD_TYPE and default to RelWithDebInfo.
if ("" STREQUAL "${CMAKE_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)

  set_property(
    CACHE CMAKE_BUILD_TYPE
    PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel
  )
endif ()

# Disable compiler extensions:
set(CMAKE_CXX_EXTENSIONS OFF)

thrust_configure_multiconfig()
thrust_find_thrust()
thrust_build_compiler_targets()
thrust_update_system_found_flags()
if (THRUST_CUDA_FOUND)
  include(cmake/ThrustCudaConfig.cmake)
endif()
thrust_build_target_list()

message(STATUS "CPP system found?  ${THRUST_CPP_FOUND}")
message(STATUS "CUDA system found? ${THRUST_CUDA_FOUND}")
message(STATUS "TBB system found?  ${THRUST_TBB_FOUND}")
message(STATUS "OMP system found?  ${THRUST_OMP_FOUND}")

if (THRUST_ENABLE_HEADER_TESTING)
  include(cmake/ThrustHeaderTesting.cmake)
endif()

# Both testing and examples use ctest
if (THRUST_ENABLE_TESTING OR THRUST_ENABLE_EXAMPLES)
  include(CTest)
  enable_testing()
endif()

if (THRUST_ENABLE_TESTING)
  add_subdirectory(testing)
endif()

if (THRUST_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

if (THRUST_ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
