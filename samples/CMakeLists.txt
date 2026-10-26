cmake_minimum_required(VERSION 3.20)

project(cccl-samples LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default arch list can be overridden with -DCMAKE_CUDA_ARCHITECTURES=...
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(
    CMAKE_CUDA_ARCHITECTURES
    75
    80
    86
    87
    89
    90
    100
    110
    120
  )
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if (ENABLE_CUDA_DEBUG)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G") # enable cuda-gdb
else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line info for debug tools
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# Add MSVC-specific flags for standard-conforming preprocessor (required for CCCL)
if (MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()

# Include installation configuration before processing samples
include(cmake/InstallSamples.cmake)

# In-tree default: point CCCL fetches at the parent CCCL checkout so
# samples using CCCL:: targets build against the surrounding repository
# instead of downloading a pinned release via CPM.
# Users building the samples tree standalone can override
# -DCCCL_SOURCE_DIR=... or leave it empty to fall back to CPM.
if (NOT DEFINED CCCL_SOURCE_DIR)
  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../cccl-version.json")
    get_filename_component(
      CCCL_SOURCE_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/.."
      ABSOLUTE
    )
    set(
      CCCL_SOURCE_DIR
      "${CCCL_SOURCE_DIR}"
      CACHE PATH
      "Path to a local CCCL checkout used by the samples."
    )
  endif()
endif()

# CCCL samples (flat layout - one add_subdirectory per sample).
# Keep these in alphabetical order as samples are added.
add_subdirectory(cubDeviceFind)
