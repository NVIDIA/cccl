enable_language(CUDA)

#
# Architecture options:
#

# Create a new arch list that only contains arches that support CDP:
if ("native" IN_LIST CMAKE_CUDA_ARCHITECTURES)
  set(THRUST_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES_NATIVE})
elseif ("all" IN_LIST CMAKE_CUDA_ARCHITECTURES)
  set(THRUST_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES_ALL})
elseif ("all-major" IN_LIST CMAKE_CUDA_ARCHITECTURES)
  set(THRUST_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR})
else()
  set(THRUST_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
endif()
set(THRUST_CUDA_ARCHITECTURES_RDC ${THRUST_CUDA_ARCHITECTURES})
list(FILTER THRUST_CUDA_ARCHITECTURES_RDC EXCLUDE REGEX "53|62|72")

message(STATUS "THRUST_CUDA_ARCHITECTURES:     ${THRUST_CUDA_ARCHITECTURES}")
message(STATUS "THRUST_CUDA_ARCHITECTURES_RDC: ${THRUST_CUDA_ARCHITECTURES_RDC}")

option(THRUST_ENABLE_RDC_TESTS "Enable tests that require separable compilation." ON)
option(THRUST_FORCE_RDC "Enable separable compilation on all targets that support it." OFF)

list(LENGTH THRUST_CUDA_ARCHITECTURES_RDC rdc_arch_count)
if (rdc_arch_count EQUAL 0)
  message(NOTICE "Disabling THRUST CDPv1 targets as no enabled architectures support it.")
  set(THRUST_ENABLE_RDC_TESTS OFF CACHE BOOL "" FORCE)
  set(THRUST_FORCE_RDC OFF CACHE BOOL "" FORCE)
endif()

#
# Clang CUDA options
#
if ("Clang" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-unknown-cuda-version -Xclang=-fcuda-allow-variadic-functions")
endif ()
