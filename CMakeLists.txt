# 3.15 is the minimum.
# 3.17 for NVC++.
# 3.18 for C++17 + CUDA.
cmake_minimum_required(VERSION 3.15)

# Determine whether libcudacxx is the top-level project or included into
# another project via add_subdirectory().
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_LIST_DIR}")
  set(LIBCUDACXX_TOPLEVEL_PROJECT ON)
else()
  set(LIBCUDACXX_TOPLEVEL_PROJECT OFF)
endif()

set(PACKAGE_NAME libcudacxx)
set(PACKAGE_VERSION 11.0)
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
project(libcudacxx NONE)

include(cmake/libcudacxxInstallRules.cmake)

if (NOT LIBCUDACXX_TOPLEVEL_PROJECT)
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

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(LLVM_PATH "${CMAKE_SOURCE_DIR}" CACHE STRING "" FORCE)

# Configuration options.
option(LIBCUDACXX_ENABLE_CUDA "Enable the CUDA language support." ON)
set(_libcudacxx_enable_static_library OFF)
if ("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVCXX")
  set(_libcudacxx_enable_static_library ON)
endif ()
option(LIBCUDACXX_ENABLE_STATIC_LIBRARY "Enable building the full C++ stdlib static library build."
    ${_libcudacxx_enable_static_library})
option(LIBCUDACXX_ENABLE_LIBCUDACXX_TESTS "Enable libcu++ tests." ON)
option(LIBCUDACXX_ENABLE_LIBCXX_TESTS "Enable upstream libc++ tests." OFF)
option(LIBCUDACXX_ENABLE_LIBCXXABI_TESTS "Enable upstream libc++abi tests." OFF)
option(LIBCUDACXX_ENABLE_LIBUNWIND_TESTS "Enable upstream libunwind tests." OFF)

# This must be done before any languages are enabled:
if (LIBCUDACXX_TOPLEVEL_PROJECT)
  include(cmake/libcudacxxCompilerHacks.cmake)
endif()

# This must appear after our Compiler Hacks or else CMake will delete the cache
# and reconfigure from scratch.
# This must also appear before the installation rules, as it is required by the
# GNUInstallDirs CMake module.
enable_language(CXX)

if (LIBCUDACXX_ENABLE_CUDA)
  enable_language(CUDA)
endif ()

set(_libcudacxx_enable_upstream_tests OFF)
if (LIBCUDACXX_ENABLE_LIBCXX_TESTS
    OR LIBCUDACXX_ENABLE_LIBCXXABI_TESTS
    OR LIBCUDACXX_ENABLE_LIBUNWIND_TESTS)
  set(_libcudacxx_enable_upstream_tests ON)
endif ()

set(_libcudacxx_enable_tests OFF)
if (_libcudacxx_enable_upstream_tests OR LIBCUDACXX_ENABLE_LIBCUDACXX_TESTS)
  set(_libcudacxx_enable_tests ON)
endif ()

if (LIBCUDACXX_ENABLE_STATIC_LIBRARY OR _libcudacxx_enable_upstream_tests)
  enable_language(C ASM)
endif ()

if (_libcudacxx_enable_tests OR LIBCUDACXX_ENABLE_STATIC_LIBRARY)
  include(FindPythonInterp)
  if (NOT PYTHONINTERP_FOUND)
    message(FATAL_ERROR
      "Failed to find python interpreter, which is required for running tests and "
      "building a libcu++ static library.")
  endif ()
endif ()

# Determine the host triple to avoid invoking `${CXX} -dumpmachine`.
include(GetHostTriple)
get_host_triple(LLVM_INFERRED_HOST_TRIPLE)

set(LLVM_HOST_TRIPLE "${LLVM_INFERRED_HOST_TRIPLE}" CACHE STRING
    "Host on which LLVM binaries will run")

# By default, we target the host, but this can be overridden at CMake
# invocation time.
set(LLVM_DEFAULT_TARGET_TRIPLE "${LLVM_HOST_TRIPLE}" CACHE STRING
  "Default target for which LLVM will generate code." )
set(TARGET_TRIPLE "${LLVM_DEFAULT_TARGET_TRIPLE}")
message(STATUS "LLVM host triple: ${LLVM_HOST_TRIPLE}")
message(STATUS "LLVM default target triple: ${LLVM_DEFAULT_TARGET_TRIPLE}")

# Determine which of the components we need to add to the build.
# By default, none.
# These variables are overriden later on if they are determined to be needed.
set(_libcudacxx_add_libcxx OFF)
set(_libcudacxx_add_libcxxabi OFF)
set(_libcudacxx_add_libunwind OFF)

# Configure options needed to build the standalone static library.
if (LIBCUDACXX_ENABLE_STATIC_LIBRARY)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "NVCXX")
    option(LIBUNWIND_ENABLE_PEDANTIC OFF)
    add_definitions(-D__USER_LABEL_PREFIX__=)
  endif()

  option(LIBUNWIND_ENABLE_STATIC "" ON)
  option(LIBUNWIND_ENABLE_SHARED "" OFF)

  #set(HAVE_LIBUNWIND ON CACHE BOOL "")
  option(LIBCXXABI_USE_LLVM_UNWINDER "" ON)
  option(LIBCXXABI_ENABLE_STATIC "" ON)
  option(LIBCXXABI_ENABLE_SHARED "" OFF)
  option(LIBCXXABI_ENABLE_STATIC_UNWINDER "" ON)
  option(LIBCXXABI_STATICALLY_LINK_UNWINDER_IN_STATIC_LIBRARY "" ON)

  option(LIBCXX_ENABLE_STATIC "" ON)
  option(LIBCXX_ENABLE_STATIC_ABI_LIBRARY "" ON)
  option(LIBCXX_ENABLE_SHARED "" OFF)
  option(LIBCXX_ENABLE_SHARED_ABI_LIBRARY "" OFF)
  option(LIBCXX_STATICALLY_LINK_ABI_IN_STATIC_LIBRARY "" ON)

  if (NOT "${LIBCXX_CXX_ABI}" STREQUAL "" AND NOT "${LIBCXX_CXX_ABI}" STREQUAL "libcxxabi")
    unset(LIBCXX_CXX_ABI CACHE)
    message(FATAL_ERROR
      "When building a standalone libcu++ static library, manually setting the "
      "C++ ABI library is not supported. Please unset the LIBCXX_CXX_ABI variable.")
  endif ()

  set(LIBCXX_CXX_ABI "libcxxabi" CACHE STRING "")
  set(LIBCXX_CXX_ABI_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/libcxxabi/include" CACHE STRING "")
  set(LIBCXX_CXX_ABI_INTREE 1)

  set(_libcudacxx_add_libcxx ON)
  set(_libcudacxx_add_libcxxabi ON)
  set(_libcudacxx_add_libunwind ON)
endif ()

if ("${LIBCXX_CXX_ABI}" STREQUAL "")
  set(LIBCXX_CXX_ABI "libstdc++" CACHE STRING "")
endif ()


# Configure options needed to run tests.
if (_libcudacxx_enable_tests)
  enable_testing()

  set(LIT_EXTRA_ARGS "" CACHE STRING "Use for additional options (e.g. -j12)")

  find_program(LLVM_DEFAULT_EXTERNAL_LIT lit)
  set(LLVM_LIT_ARGS "-sv ${LIT_EXTRA_ARGS}")

  if (LIBCUDACXX_ENABLE_LIBCUDACXX_TESTS)
    set(LIBCUDACXX_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

    add_subdirectory(test)
  endif ()

  if (LIBCUDACXX_ENABLE_LIBCXX_TESTS)
    option(LIBCXX_INCLUDE_TESTS "" ON)

    set(_libcudacxx_add_libcxx ON)
  endif ()

  if (LIBCUDACXX_ENABLE_LIBCXXABI_TESTS)
    message(FATAL_ERROR "Running libc++abi tests is not supported yet.")

    set(_libcudacxx_add_libcxxabi ON)
  endif ()

  if (LIBCUDACXX_ENABLE_LIBUNWIND_TESTS)
    message(FATAL_ERROR "Running libunwind tests is not supported yet.")

    set(_libcudacxx_add_libunwind ON)
  endif ()
endif ()

# Configure the common options for the subdirections and add them.
if (_libcudacxx_add_libunwind)
  add_subdirectory(libunwind)
endif ()

if (_libcudacxx_add_libcxxabi)
  add_subdirectory(libcxxabi)
endif ()

if (_libcudacxx_add_libcxx)
  option(LIBCXX_INCLUDE_BENCHMARKS "" OFF)
  option(LIBCXX_ENABLE_EXPERIMENTAL_LIBRARY "" OFF)

  set(LIBCXX_HAVE_CXX_ATOMICS_WITHOUT_LIB ON)

  add_subdirectory(libcxx)
endif ()

# Add a global check rule now that all subdirectories have been traversed
# and we know the total set of lit testsuites.
if (_libcudacxx_enable_tests)
  include(AddLLVM)

  get_property(LLVM_LIT_TESTSUITES GLOBAL PROPERTY LLVM_LIT_TESTSUITES)
  get_property(LLVM_LIT_PARAMS GLOBAL PROPERTY LLVM_LIT_PARAMS)
  get_property(LLVM_LIT_DEPENDS GLOBAL PROPERTY LLVM_LIT_DEPENDS)
  get_property(LLVM_LIT_EXTRA_ARGS GLOBAL PROPERTY LLVM_LIT_EXTRA_ARGS)
  get_property(LLVM_ADDITIONAL_TEST_TARGETS
               GLOBAL PROPERTY LLVM_ADDITIONAL_TEST_TARGETS)
  get_property(LLVM_ADDITIONAL_TEST_DEPENDS
               GLOBAL PROPERTY LLVM_ADDITIONAL_TEST_DEPENDS)
  add_lit_target(check-all
    "Running all regression tests"
    ${LLVM_LIT_TESTSUITES}
    PARAMS ${LLVM_LIT_PARAMS}
    DEPENDS ${LLVM_LIT_DEPENDS} ${LLVM_ADDITIONAL_TEST_TARGETS}
    ARGS ${LLVM_LIT_EXTRA_ARGS}
  )
endif ()

