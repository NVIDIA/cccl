# 3.15 is the minimum for including the project with add_subdirectory.
# 3.21 is the minimum for the developer build.
cmake_minimum_required(VERSION 3.15)

# Determine whether CCCL is the top-level project or included into
# another project via add_subdirectory()
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_LIST_DIR}")
  set(CCCL_TOPLEVEL_PROJECT ON)
endif()

# TODO find_package for add_subdir usecases, use components for subprojects.

project(CCCL NONE)

# Optionally include installation rules for non-top-level builds:
option(CCCL_ENABLE_INSTALL_RULES "Enable installation of CCCL." ${CCCL_TOPLEVEL_PROJECT})
if (CCCL_ENABLE_INSTALL_RULES)
  include(cmake/CCCLInstallRules.cmake)
endif()

# Support adding Thrust to a parent project via add_subdirectory.
# See examples/cmake/add_subdir/CMakeLists.txt for details.
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

include(CTest)
enable_testing()

if (CCCL_ENABLE_LIBCUDACXX)
  add_subdirectory(libcudacxx)
endif()

if (CCCL_ENABLE_CUB)
  add_subdirectory(cub)
endif()

if (CCCL_ENABLE_THRUST)
  add_subdirectory(thrust)
endif()

if (CCCL_ENABLE_TESTING)
  add_subdirectory(test)
endif()
