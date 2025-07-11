enable_language(CUDA)

#
# Architecture options:
#

option(CUB_ENABLE_RDC_TESTS "Enable tests that require separable compilation." ON)
option(CUB_FORCE_RDC "Enable separable compilation on all targets that support it." OFF)

#
# Clang CUDA options
#
if ("Clang" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-unknown-cuda-version -Xclang=-fcuda-allow-variadic-functions")
endif ()
