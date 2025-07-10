enable_language(CUDA)

#
# Architecture options:
#

# Since we have to filter the arch list based on target features, we don't
# currently support the convenience arch flags:
if ("all" IN_LIST CMAKE_CUDA_ARCHITECTURES OR
    "all-major" IN_LIST CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR
    "The Thrust dev build requires an explicit list of architectures (including 'native') in CMAKE_CUDA_ARCHITECTURES. "
    "The convenience flags of 'all' and 'all-major' are not supported.\n"
    "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Replace 'native' with the actual compute capabilities by querying nvidia-smi.
if (CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
  execute_process(
    COMMAND "nvidia-smi" "--query-gpu=compute_cap" "--format=csv,noheader"
    OUTPUT_VARIABLE native_compute_caps
    RESULT_VARIABLE exit_code
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if (exit_code)
    message(FATAL_ERROR
      "Failed to query native compute capabilities using nvidia-smi. "
      "Please remove 'native' from CMAKE_CUDA_ARCHITECTURES."
    )
  endif()
  # nvidia-smi returns for example "12.0\n8.6"
  string(REPLACE "." "" native_compute_caps "${native_compute_caps}")
  string(REPLACE "\n" "-real;" native_compute_caps "${native_compute_caps}")
  string(APPEND native_compute_caps "-real")
  message(STATUS "Expanded 'native' in CMAKE_CUDA_ARCHITECTURES to '${native_compute_caps}'")
  set(CMAKE_CUDA_ARCHITECTURES ${native_compute_caps})
endif()

# Create a new arch list that only contains arches that support CDP:
set(THRUST_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
set(THRUST_CUDA_ARCHITECTURES_RDC ${THRUST_CUDA_ARCHITECTURES})
list(FILTER THRUST_CUDA_ARCHITECTURES_RDC EXCLUDE REGEX "53|62|72|101")

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
