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
endif()

# We require a higher cmake version for dev builds
if (CCCL_TOPLEVEL_PROJECT)
  cmake_minimum_required(VERSION 3.21)
endif()

option(CCCL_ENABLE_LIBCUDACXX "Enable the libcu++ developer build." ${CCCL_TOPLEVEL_PROJECT})
option(CCCL_ENABLE_CUB "Enable the CUB developer build." ${CCCL_TOPLEVEL_PROJECT})
option(CCCL_ENABLE_THRUST "Enable the Thrust developer build." ${CCCL_TOPLEVEL_PROJECT})
option(CCCL_ENABLE_TESTING "Enable CUDA C++ Core Library tests." ${CCCL_TOPLEVEL_PROJECT})
option(CCCL_ENABLE_EXAMPLES "Enable CUDA C++ Core Library examples." ${CCCL_TOPLEVEL_PROJECT})
option(CCCL_ENABLE_BENCHMARKS "Enable CUDA C++ Core Library benchmarks." OFF)
option(CCCL_ENABLE_C "Enable CUDA C Core Library." OFF)

option(CCCL_ENABLE_UNSTABLE "Enable targets and developer build options for unstable projects." OFF)

if (CCCL_ENABLE_UNSTABLE)
  option(CCCL_ENABLE_CUDAX "Enable the CUDA Experimental developer build." ON)
endif()

option(CCCL_DISABLE_EXCEPTIONS "Disable use of exceptions within CCCL libraries." OFF)
if (CCCL_DISABLE_EXCEPTIONS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCCCL_DISABLE_EXCEPTIONS")
endif()

include(CTest)
enable_testing()

if (CCCL_TOPLEVEL_PROJECT)
  include(cmake/AppendOptionIfAvailable.cmake)
  include(cmake/CCCLUtilities.cmake) # include this before other CCCL helpers

  include(cmake/CCCLBuildCompilerTargets.cmake)
  include(cmake/CCCLClangdCompileInfo.cmake)
  include(cmake/CCCLConfigureTarget.cmake)

  cccl_build_compiler_targets()
endif()

if (CCCL_ENABLE_LIBCUDACXX)
  set(LIBCUDACXX_TOPLEVEL_PROJECT ${CCCL_TOPLEVEL_PROJECT})
endif()

if (CCCL_ENABLE_CUB)
  set(CUB_TOPLEVEL_PROJECT ${CCCL_TOPLEVEL_PROJECT})
endif()

if (CCCL_ENABLE_THRUST)
  set(THRUST_TOPLEVEL_PROJECT ${CCCL_TOPLEVEL_PROJECT})
endif()

if (CCCL_ENABLE_CUDAX)
  set(cudax_TOPLEVEL_PROJECT ${CCCL_TOPLEVEL_PROJECT})
endif()

add_subdirectory(libcudacxx)
add_subdirectory(cub)
add_subdirectory(thrust)

if (CCCL_ENABLE_UNSTABLE)
  add_subdirectory(cudax)
endif()

if (CCCL_ENABLE_C)
  add_subdirectory(c)
endif()

if (CCCL_ENABLE_TESTING)
  add_subdirectory(test)
endif()

if (CCCL_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

# Must stay at the end of this file.
include(cmake/CCCLHideThirdPartyOptions.cmake)
