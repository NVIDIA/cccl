# 3.15 is the minimum for including the project with add_subdirectory.
# 3.21 is the minimum for the developer build.
cmake_minimum_required(VERSION 3.15)

# This must be done before any languages are enabled:
if (CCCL_ENABLE_CUDAX)
  cmake_minimum_required(VERSION 3.21)
endif()

project(cudax LANGUAGES CXX)

if (NOT CCCL_ENABLE_CUDAX)
  include(cmake/cudaxAddSubdir.cmake)
  return()
endif()

# Enable CUDA only when required for the developer build, see #2609
enable_language(CUDA)

option(cudax_ENABLE_HEADER_TESTING "Test that CUDA Experimental's public headers compile." ON)
option(cudax_ENABLE_TESTING "Build CUDA Experimental's tests." ON)
option(cudax_ENABLE_EXAMPLES "Build CUDA Experimental's examples." ON)
option(cudax_ENABLE_CUDASTF "Enable CUDASTF subproject" ON)
option(cudax_ENABLE_CUDASTF_CODE_GENERATION "Enable code generation using STF's parallel_for or launch with CUDA compiler." ON)
option(cudax_ENABLE_CUDASTF_BOUNDSCHECK "Enable bounds checks for STF targets. Requires debug build." OFF)
option(cudax_ENABLE_CUDASTF_MATHLIBS "Enable STF tests/examples that use cublas/cusolver." OFF)

if (cudax_ENABLE_CUDASTF_BOUNDSCHECK AND
    NOT CMAKE_BUILD_TYPE MATCHES "Debug" AND NOT CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
  message(FATAL_ERROR "cudax_ENABLE_CUDASTF_BOUNDSCHECK requires a Debug build.")
endif()

include(cmake/cudaxBuildCompilerTargets.cmake)
include(cmake/cudaxBuildTargetList.cmake)
if (cudax_ENABLE_CUDASTF)
  include(cmake/cudaxSTFConfigureTarget.cmake)
endif()

cudax_build_compiler_targets()
cudax_build_target_list()

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
