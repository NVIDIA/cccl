enable_language(CUDA)

#
# Architecture options:
#

# Create a new arch list that only contains arches that support CDP:
if ("native" IN_LIST CMAKE_CUDA_ARCHITECTURES)
  set(CUB_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES_NATIVE})
elseif ("all" IN_LIST CMAKE_CUDA_ARCHITECTURES)
  set(CUB_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES_ALL})
elseif ("all-major" IN_LIST CMAKE_CUDA_ARCHITECTURES)
  set(CUB_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR})
else()
  set(CUB_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
endif()
set(CUB_CUDA_ARCHITECTURES_RDC ${CUB_CUDA_ARCHITECTURES})
list(FILTER CUB_CUDA_ARCHITECTURES_RDC EXCLUDE REGEX "53|62|72")

message(STATUS "CUB_CUDA_ARCHITECTURES:     ${CUB_CUDA_ARCHITECTURES}")
message(STATUS "CUB_CUDA_ARCHITECTURES_RDC: ${CUB_CUDA_ARCHITECTURES_RDC}")

if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  # Currently, there are linkage issues caused by bugs in interaction between MSBuild and CMake object libraries
  # that take place with -rdc builds. Changing the default for now.
  option(CUB_ENABLE_RDC_TESTS "Enable tests that require separable compilation." OFF)
else()
  option(CUB_ENABLE_RDC_TESTS "Enable tests that require separable compilation." ON)
endif()

option(CUB_FORCE_RDC "Enable separable compilation on all targets that support it." OFF)

list(LENGTH CUB_CUDA_ARCHITECTURES_RDC rdc_arch_count)
if (rdc_arch_count EQUAL 0)
  message(NOTICE "Disabling CUB CDPv1 targets as no enabled architectures support it.")
  set(CUB_ENABLE_RDC_TESTS OFF CACHE BOOL "" FORCE)
  set(CUB_FORCE_RDC OFF CACHE BOOL "" FORCE)
endif()

#
# Clang CUDA options
#
if ("Clang" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-unknown-cuda-version -Xclang=-fcuda-allow-variadic-functions")
endif ()
