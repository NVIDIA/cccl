# 3.15 is the minimum for including the project with add_subdirectory.
# 3.18 for C++17 + CUDA and clang-cuda support.
# 3.21 is the minimum for the developer build.
cmake_minimum_required(VERSION 3.15)

# Determine whether libcudacxx is the top-level project or included into
# another project via add_subdirectory().
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_LIST_DIR}")
  set(LIBCUDACXX_TOPLEVEL_PROJECT ON)
endif()

if (LIBCUDACXX_TOPLEVEL_PROJECT)
  cmake_minimum_required(VERSION 3.21)
endif()

set(PACKAGE_NAME libcudacxx)
set(PACKAGE_VERSION 11.0)
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
project(libcudacxx LANGUAGES CXX)

if (NOT LIBCUDACXX_TOPLEVEL_PROJECT)
  include(cmake/libcudacxxAddSubdir.cmake)
  return()
endif()

# Must be called in root CMakeLists.txt
include(CTest)
enable_testing()

# Add codegen module
option(libcudacxx_ENABLE_CODEGEN "Enable libcudacxx's atomics backend codegen and tests." OFF)
if (libcudacxx_ENABLE_CODEGEN)
  add_subdirectory(codegen)
endif()

set(CMAKE_MODULE_PATH "${libcudacxx_SOURCE_DIR}/cmake")
set(LLVM_PATH "${libcudacxx_SOURCE_DIR}" CACHE STRING "" FORCE)

# Configuration options.
option(LIBCUDACXX_ENABLE_CUDA "Enable the CUDA language support." ON)
if (LIBCUDACXX_ENABLE_CUDA)
  enable_language(CUDA)
endif ()

option(LIBCUDACXX_ENABLE_LIBCUDACXX_TESTS "Enable libcu++ tests." ON)
if (LIBCUDACXX_ENABLE_LIBCUDACXX_TESTS)
  enable_testing()

  include(FindPythonInterp)
  if (NOT PYTHONINTERP_FOUND)
    message(FATAL_ERROR
      "Failed to find python interpreter, which is required for running tests and "
      "building a libcu++ static library.")
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

  set(LIT_EXTRA_ARGS "" CACHE STRING "Use for additional options (e.g. -j12)")
  find_program(LLVM_DEFAULT_EXTERNAL_LIT lit)
  set(LLVM_LIT_ARGS "-sv ${LIT_EXTRA_ARGS}")

  set(LIBCUDACXX_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  add_subdirectory(test)

  include(AddLLVM)
  get_property(LLVM_LIT_TESTSUITES          GLOBAL PROPERTY LLVM_LIT_TESTSUITES)
  get_property(LLVM_LIT_PARAMS              GLOBAL PROPERTY LLVM_LIT_PARAMS)
  get_property(LLVM_LIT_DEPENDS             GLOBAL PROPERTY LLVM_LIT_DEPENDS)
  get_property(LLVM_LIT_EXTRA_ARGS          GLOBAL PROPERTY LLVM_LIT_EXTRA_ARGS)
  get_property(LLVM_ADDITIONAL_TEST_TARGETS GLOBAL PROPERTY LLVM_ADDITIONAL_TEST_TARGETS)
  get_property(LLVM_ADDITIONAL_TEST_DEPENDS GLOBAL PROPERTY LLVM_ADDITIONAL_TEST_DEPENDS)
  add_lit_target(check-all
    "Running all regression tests"
    ${LLVM_LIT_TESTSUITES}
    PARAMS ${LLVM_LIT_PARAMS}
    DEPENDS ${LLVM_LIT_DEPENDS} ${LLVM_ADDITIONAL_TEST_TARGETS}
    ARGS ${LLVM_LIT_EXTRA_ARGS}
  )
endif ()
