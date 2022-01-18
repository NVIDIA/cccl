cmake_minimum_required(VERSION 3.8)
project(libcudacxx CXX)

# Determine whether libcudacxx is the top-level project or included into
# another project via add_subdirectory().
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_LIST_DIR}")
  set(libcudacxx_TOPLEVEL_PROJECT ON)
else()
  set(libcudacxx_TOPLEVEL_PROJECT OFF)
endif()

include(cmake/libcudacxxInstallRules.cmake)

if (NOT libcudacxx_TOPLEVEL_PROJECT)
  include(cmake/libcudacxxAddSubdir.cmake)
  return()
endif()

# Note that this currently returns and skips the rest of the build
# system.
option(libcudacxx_ENABLE_CMAKE_TESTS "Enable ctest-based testing." OFF)
if (libcudacxx_ENABLE_CMAKE_TESTS)
  # Might be able to lower this, but would need to do some testing:
  cmake_minimum_required(VERSION 3.20.1)
  include(CTest)
  enable_testing() # Must be called in root CMakeLists.txt
  add_subdirectory(cmake/test/)
  return()
endif()

set(PACKAGE_NAME libcudacxx)
set(PACKAGE_VERSION 11.0)
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")

if (NOT CMAKE_C_COMPILER)
  set(CMAKE_C_COMPILER ${CMAKE_CXX_COMPILER})
  set(CMAKE_C_COMPILER_ARG1 ${CMAKE_CXX_COMPILER_ARG1})
endif ()

enable_language(C)

set(LIBCXX_HIGHEST_COMPUTE_ARCH 80)
set(LIBCXX_KNOWN_COMPUTE_ARCHS 35 50 52 53 60 61 62 70 72 75 80)

option(LIBCXX_DISABLE_ARCH_BY_DEFAULT "If ON, then all CUDA architectures are disabled on the initial CMake run." OFF)
set(OPTION_INIT ON)
if (LIBCXX_DISABLE_ARCH_BY_DEFAULT)
  set(OPTION_INIT OFF)
endif ()
if (NOT ${LIBCXX_HIGHEST_COMPUTE_ARCH} IN_LIST LIBCXX_KNOWN_COMPUTE_ARCHS)
  message(FATAL_ERROR "When changing the highest compute version, don't forget to add it to the list!")
endif ()

foreach (COMPUTE_ARCH IN LISTS LIBCXX_KNOWN_COMPUTE_ARCHS)
  option(LIBCXX_ENABLE_COMPUTE_${COMPUTE_ARCH} "Enable code generation for tests for sm_${COMPUTE_ARCH}" ${OPTION_INIT})
  if (LIBCXX_ENABLE_COMPUTE_${COMPUTE_ARCH})
    set(LIBCXX_COMPUTE_ARCHS "${LIBCXX_COMPUTE_ARCHS} ${COMPUTE_ARCH}")
    set(COMPUTE_MESSAGE "${COMPUTE_MESSAGE} sm_${COMPUTE_ARCH}")
  endif ()
endforeach ()

option(LIBCXX_ENABLE_COMPUTE_FUTURE "Enable code generation for tests for compute_${LIBCXX_HIGHEST_COMPUTE_ARCH}" ${OPTION_INIT})
if (LIBCXX_ENABLE_COMPUTE_FUTURE)
  set(COMPUTE_MESSAGE "${COMPUTE_MESSAGE} compute_${LIBCXX_HIGHEST_COMPUTE_ARCH}")
endif ()

message(STATUS "Enabled CUDA architectures:${COMPUTE_MESSAGE}")

# additionally to the above options for controlling the enabled architectures,
# we configure a project that will be used by the perform_test to find the SM architecture
# of its first visible device

# we do that here, because here it is easiest to determine what the _actual_ compiler that will
# produce an executable to run on the host is (NVRTC makes this harder than it'd be otherwise)

# this is used in automated and semi-automated testing

if (NOT LIBCXX_TEST_WITH_NVRTC)
    set(LIBCXX_FORCE_INCLUDE "-include ${CMAKE_SOURCE_DIR}/test/force_include.h")
endif()

# TODO: this has to be reconfigurable with other options
# this has to land in cache, but has to be forced, and controlled by other variables
set(LIBCXX_TEST_COMPILER_FLAGS
    "${LIBCXX_FORCE_INCLUDE} -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_SOURCE_DIR}/../cuda/tools/cooperative_groups -I${CMAKE_SOURCE_DIR}/../cuda/tools/libcudacxxext"
    CACHE STRING "Flags for libcxx testing." FORCE)

foreach (PATH IN LISTS LIBCXX_TEST_LD_LIBRARY_PATHS)
  if (LIBCXX_TEST_RPATH_FLAGS)
    set(PREVIOUS "${LIBCXX_TEST_RPATH_FLAGS} ")
  endif ()
  set(LIBCXX_TEST_RPATH_FLAGS "${PREVIOUS}-Xcompiler \\\\\\\"-Wl,-rpath,${PATH}\\\\\\\"")
endforeach ()

set(LIBCXX_TEST_LINKER_FLAGS
    "${LIBCXX_TEST_RPATH_FLAGS}"
    CACHE STRING "Flags for libcxx testing.")

set(LIBCXX_ENABLE_EXPERIMENTAL_LIBRARY OFF CACHE BOOL "")
set(LIBCXX_ENABLE_EXCEPTIONS OFF CACHE BOOL "")
set(LIBCXX_ENABLE_FILESYSTEM OFF CACHE BOOL "")
set(LIBCXX_ENABLE_RTTI OFF CACHE BOOL "")
set(LIBCXX_ENABLE_SHARED OFF CACHE BOOL "")
set(LIBCXX_INCLUDE_TESTS ON CACHE BOOL "")
set(LIBCXX_INCLUDE_BENCHMARKS OFF CACHE BOOL "")
set(LIBCXX_HAS_ATOMIC_LIB OFF CACHE BOOL "")
set(LIBCXX_CXX_ABI "none" CACHE STRING "")

set(LIBCXX_SOURCE_DIR ${CMAKE_SOURCE_DIR} CACHE STRING "")

add_subdirectory(libcxx)
