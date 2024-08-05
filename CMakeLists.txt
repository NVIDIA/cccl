# 3.15 is the minimum for including the project with add_subdirectory.
# 3.21 is the minimum for the developer build.
cmake_minimum_required(VERSION 3.15)

# sccache cannot handle the -Fd option generating pdb files:
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
endif()

# Determine whether CCCL is the top-level project or included into
# another project via add_subdirectory()
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_LIST_DIR}")
  set(CCCL_TOPLEVEL_PROJECT ON)
endif()

# Enable CXX so CMake can configure install paths
project(CCCL LANGUAGES CXX)

# Optionally include installation rules for non-top-level builds:
option(CCCL_ENABLE_INSTALL_RULES "Enable installation of CCCL." ${CCCL_TOPLEVEL_PROJECT})
if (CCCL_ENABLE_INSTALL_RULES)
  include(cmake/CCCLInstallRules.cmake)
endif()

# Support adding CCCL to a parent project via add_subdirectory.
if (NOT CCCL_TOPLEVEL_PROJECT)
  include(cmake/CCCLAddSubdir.cmake)
  return()
endif()

# We require a higher cmake version for dev builds
cmake_minimum_required(VERSION 3.21)

option(CCCL_ENABLE_LIBCUDACXX "Enable the libcu++ developer build." ON)
option(CCCL_ENABLE_CUB "Enable the CUB developer build." ON)
option(CCCL_ENABLE_THRUST "Enable the Thrust developer build." ON)
option(CCCL_ENABLE_TESTING "Enable CUDA C++ Core Library tests." ON)
option(CCCL_ENABLE_EXAMPLES "Enable CUDA C++ Core Library examples." ON)
option(CCCL_ENABLE_BENCHMARKS "Enable CUDA C++ Core Library benchmarks." OFF)

option(CCCL_ENABLE_UNSTABLE "Enable targets and developer build options for unstable projects." OFF)

if (CCCL_ENABLE_UNSTABLE)
  option(CCCL_ENABLE_CUDAX "Enable the CUDA Experimental developer build." ON)
endif()


include(CTest)
enable_testing()

include(cmake/CCCLUtilities.cmake) # include this first
include(cmake/CCCLClangdCompileInfo.cmake)

if (CCCL_ENABLE_LIBCUDACXX)
  set(LIBCUDACXX_TOPLEVEL_PROJECT ON)
endif()

if (CCCL_ENABLE_CUB)
  set(CUB_TOPLEVEL_PROJECT ON)
endif()

if (CCCL_ENABLE_THRUST)
  set(THRUST_TOPLEVEL_PROJECT ON)
endif()

if (CCCL_ENABLE_CUDAX)
  set(cudax_TOPLEVEL_PROJECT ON)
endif()

add_subdirectory(libcudacxx)
add_subdirectory(cub)
add_subdirectory(thrust)

if (CCCL_ENABLE_UNSTABLE)
  add_subdirectory(cudax)
endif()

if (CCCL_ENABLE_TESTING)
  add_subdirectory(test)
endif()

if (CCCL_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

# Must stay at the end of this file.
include(cmake/CCCLHideThirdPartyOptions.cmake)
