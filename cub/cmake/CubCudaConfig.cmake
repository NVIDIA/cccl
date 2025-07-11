enable_language(CUDA)

#
# Architecture options:
#

# TODO(bgruber): is this still true?
if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  # Currently, there are linkage issues caused by bugs in interaction between MSBuild and CMake object libraries
  # that take place with -rdc builds. Changing the default for now.
  option(CUB_ENABLE_RDC_TESTS "Enable tests that require separable compilation." OFF)
else()
  option(CUB_ENABLE_RDC_TESTS "Enable tests that require separable compilation." ON)
endif()

option(CUB_FORCE_RDC "Enable separable compilation on all targets that support it." OFF)

#
# Clang CUDA options
#
if ("Clang" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-unknown-cuda-version -Xclang=-fcuda-allow-variadic-functions")
endif ()
