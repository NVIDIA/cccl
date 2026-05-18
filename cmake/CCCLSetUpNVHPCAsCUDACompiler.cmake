# Sets up NVHPC as the CUDA compiler. Must be done after enabling C++ language and before enabling CUDA language.
#
# Warning: This is an internal-only feature used for cub testing for stdpar.

macro(_cccl_set_up_nvhpc_as_cuda_compiler)
  list(APPEND CMAKE_MODULE_PATH "${CCCL_SOURCE_DIR}/cmake/nvhpc-cuda-compiler")

  if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "NVHPC")
    message(
      FATAL_ERROR
      "NVHPC as the CUDA compiler must be used together with as the NVHPC C++ compiler"
    )
  endif()

  set(
    CMAKE_CUDA_COMPILER
    "${CMAKE_CXX_COMPILER}"
    CACHE FILEPATH
    "CUDA compiler"
    FORCE
  )
  set(CMAKE_CUDA_COMPILER_ID "${CMAKE_CXX_COMPILER_ID}")
  set(CMAKE_CUDA_COMPILER_VERSION "${CMAKE_CXX_COMPILER_VERSION}")
  set(CMAKE_CUDA_COMPILER_ID_RUN TRUE)
  set(CMAKE_CUDA_COMPILER_FORCED TRUE)
  set(CMAKE_CUDA_COMPILER_WORKS TRUE)
  set(
    CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT
    "${CMAKE_CXX_STANDARD_COMPUTED_DEFAULT}"
  )
  set(
    CMAKE_CUDA_EXTENSIONS_COMPUTED_DEFAULT
    "${CMAKE_CXX_EXTENSIONS_COMPUTED_DEFAULT}"
  )
endmacro()
