if (NOT CCCL_ENABLE_CUDAX)
  include(cmake/cudaxAddSubdir.cmake)
  return()
endif()

cmake_minimum_required(VERSION 3.21)
project(cudax LANGUAGES CXX CUDA)

option(
  cudax_ENABLE_HEADER_TESTING
  "Test that CUDA Experimental's public headers compile."
  ON
)
option(cudax_ENABLE_TESTING "Build CUDA Experimental's tests." ON)
option(cudax_ENABLE_EXAMPLES "Build CUDA Experimental's examples." ON)
option(cudax_ENABLE_CUDASTF "Enable CUDASTF subproject" ON)
option(
  cudax_ENABLE_CUDASTF_CODE_GENERATION
  "Enable code generation using STF's parallel_for or launch with CUDA compiler."
  ON
)
option(
  cudax_ENABLE_CUDASTF_BOUNDSCHECK
  "Enable bounds checks for STF targets. Requires debug build."
  OFF
)
option(
  cudax_ENABLE_CUDASTF_MATHLIBS
  "Enable STF tests/examples that use cublas/cusolver."
  OFF
)
option(cudax_ENABLE_CUFILE "Enable cuFile in CUDA Experimental" ON)

if (cudax_ENABLE_CUFILE)
  if (WIN32)
    message(FATAL_ERROR "cuFile is not available on Windows.")
  endif()

  if (CMAKE_VERSION VERSION_LESS "3.25.0")
    message(
      FATAL_ERROR
      "cuFile is not available before cmake 3.25.0, please, use newer cmake."
    )
  endif()

  find_package(CUDAToolkit REQUIRED)

  if (CUDAToolkit_VERSION VERSION_LESS "12.9.0")
    message(FATAL_ERROR "cuFile support requires at least CUDA 12.9.")
  endif()

  if (NOT TARGET CUDA::cuFile)
    message(
      FATAL_ERROR
      "CUDA::cuFile target required for requested cuFile support was not found."
    )
  endif()
endif()

if (
  cudax_ENABLE_CUDASTF_BOUNDSCHECK
  AND NOT CMAKE_BUILD_TYPE MATCHES "Debug"
  AND NOT CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo"
)
  message(
    FATAL_ERROR
    "cudax_ENABLE_CUDASTF_BOUNDSCHECK requires a Debug build."
  )
endif()

include(cmake/cudaxBuildCompilerTargets.cmake)
if (cudax_ENABLE_CUDASTF)
  include(cmake/cudaxSTFConfigureTarget.cmake)
endif()

if (cudax_ENABLE_HEADER_TESTING)
  include(cmake/cudaxHeaderTesting.cmake)
endif()

if (cudax_ENABLE_TESTING)
  add_subdirectory(test)
endif()

if (cudax_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

if (CCCL_ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
