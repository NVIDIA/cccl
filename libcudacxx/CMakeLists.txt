if (NOT CCCL_ENABLE_LIBCUDACXX)
  include(cmake/libcudacxxAddSubdir.cmake)
  return()
endif()

cmake_minimum_required(VERSION 3.21)
set(PACKAGE_NAME libcudacxx)
set(PACKAGE_VERSION 11.0)
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
project(libcudacxx LANGUAGES CXX)

# Add codegen module
option(
  libcudacxx_ENABLE_CODEGEN
  "Enable libcudacxx's atomics backend codegen and tests."
  OFF
)
if (libcudacxx_ENABLE_CODEGEN)
  add_subdirectory(codegen)
endif()

set(CMAKE_MODULE_PATH "${libcudacxx_SOURCE_DIR}/cmake")
set(LLVM_PATH "${libcudacxx_SOURCE_DIR}" CACHE STRING "" FORCE)

# Configuration options.
option(LIBCUDACXX_ENABLE_CUDA "Enable the CUDA language support." ON)
if (LIBCUDACXX_ENABLE_CUDA)
  enable_language(CUDA)
endif()

option(LIBCUDACXX_ENABLE_LIBCUDACXX_TESTS "Enable libcu++ tests." ON)
if (LIBCUDACXX_ENABLE_LIBCUDACXX_TESTS)
  enable_testing()

  # Create the compiler targets with the common compile flags
  include(cmake/LibcudacxxBuildCompilerTargets.cmake)
  libcudacxx_build_compiler_targets()

  # Test all public and internal headers
  include(cmake/LibcudacxxInternalHeaderTesting.cmake)
  include(cmake/LibcudacxxPublicHeaderTesting.cmake)
  include(cmake/LibcudacxxPublicHeaderTestingHost.cmake)

  add_subdirectory(test)
endif()
