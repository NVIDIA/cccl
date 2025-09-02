# 3.15 is the minimum for including the project with add_subdirectory.
# 3.18 for C++17 + CUDA and clang-cuda support.
# 3.21 is the minimum for the developer build.
cmake_minimum_required(VERSION 3.15)

if (CCCL_ENABLE_LIBCUDACXX)
  cmake_minimum_required(VERSION 3.21)
endif()

set(PACKAGE_NAME libcudacxx)
set(PACKAGE_VERSION 11.0)
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
project(libcudacxx LANGUAGES CXX)

if (NOT CCCL_ENABLE_LIBCUDACXX)
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
if (NOT LIBCUDACXX_ENABLE_LIBCUDACXX_TESTS)
  return()
endif()

enable_testing()

# Create the compiler targets with the common compile flags
include(cmake/LibcudacxxBuildCompilerTargets.cmake)
libcudacxx_build_compiler_targets()

# Test all public and internal headers
include(cmake/LibcudacxxInternalHeaderTesting.cmake)
include(cmake/LibcudacxxPublicHeaderTesting.cmake)
include(cmake/LibcudacxxPublicHeaderTestingHost.cmake)

find_package (Python COMPONENTS Interpreter)
if (NOT Python_Interpreter_FOUND)
  message(FATAL_ERROR
    "Failed to find python interpreter, which is required for running tests and building a libcu++ static library.")
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

add_subdirectory(test)
