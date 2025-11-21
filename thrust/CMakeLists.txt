if (NOT CCCL_ENABLE_THRUST)
  include(cmake/ThrustAddSubdir.cmake)
  return()
endif()

cmake_minimum_required(VERSION 3.21)
project(Thrust LANGUAGES CXX) # CUDA enabled later only if needed

option(
  THRUST_ENABLE_HEADER_TESTING
  "Test that all public headers compile."
  "ON"
)
option(THRUST_ENABLE_TESTING "Build Thrust testing suite." "ON")
option(THRUST_ENABLE_EXAMPLES "Build Thrust examples." "ON")

# Allow the user to optionally select offset type dispatch to fixed 32 or 64 bit types
set(
  THRUST_DISPATCH_TYPE
  "Dynamic"
  CACHE STRING
  "Select Thrust offset type dispatch."
)
set_property(
  CACHE THRUST_DISPATCH_TYPE
  PROPERTY STRINGS "Dynamic" "Force32bit" "Force64bit"
)

#include first:
include(cmake/ThrustUtilities.cmake)

include(cmake/ThrustBuildCompilerTargets.cmake)
include(cmake/ThrustBuildTargetList.cmake)
include(cmake/ThrustFindThrust.cmake)
include(cmake/ThrustMultiConfig.cmake)

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

if (THRUST_ENABLE_TESTING)
  add_subdirectory(testing)
endif()

if (THRUST_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

if (CCCL_ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
