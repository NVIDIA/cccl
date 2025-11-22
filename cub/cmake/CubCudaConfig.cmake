#
# Architecture options:
#

option(
  CUB_ENABLE_RDC_TESTS
  "Enable tests that require separable compilation."
  ON
)
option(
  CUB_FORCE_RDC
  "Enable separable compilation on all targets that support it."
  OFF
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  if (CUB_ENABLE_RDC_TESTS)
    if ("${CMAKE_VERSION}" VERSION_LESS 3.27.5)
      # https://gitlab.kitware.com/cmake/cmake/-/merge_requests/8794
      message(
        WARNING
        "CMake 3.27.5 or newer is required to enable RDC tests in Visual Studio."
      )
      cmake_minimum_required(VERSION 3.27.5)
    endif()
  endif()
endif()
