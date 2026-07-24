set(_cccl_cpm_file "${CMAKE_CURRENT_LIST_DIR}/CPM.cmake")
set(_cccl_find_module_dir "${CMAKE_CURRENT_LIST_DIR}/find_modules")

macro(cccl_get_boost)
  include("${_cccl_cpm_file}")
  CPMAddPackage(
    NAME Boost
    GITHUB_REPOSITORY boostorg/boost
    GIT_TAG "boost-1.83.0"
    EXCLUDE_FROM_ALL TRUE
    SYSTEM TRUE
    GIT_SHALLOW TRUE
    # Boost requests compatibility with obsolete CMake versions. Disable warning:
    OPTIONS "CMAKE_POLICY_VERSION_MINIMUM 3.5"
  )
endmacro()

# The CCCL Catch2Helper library:
macro(cccl_get_c2h)
  if (NOT TARGET cccl.c2h)
    add_subdirectory("${CCCL_SOURCE_DIR}/c2h" "${CCCL_BINARY_DIR}/c2h")
  endif()
endmacro()

macro(cccl_get_catch2)
  include("${_cccl_cpm_file}")
  CPMAddPackage("gh:catchorg/Catch2@3.12.0")
endmacro()

macro(cccl_get_cccl)
  find_package(
    CCCL
    CONFIG
    REQUIRED
    NO_DEFAULT_PATH # Only check the explicit HINTS below:
    HINTS "${CCCL_SOURCE_DIR}/lib/cmake/cccl/"
  )
endmacro()

macro(cccl_get_cub)
  find_package(
    CUB
    CONFIG
    REQUIRED
    NO_DEFAULT_PATH # Only check the explicit HINTS below:
    HINTS "${CCCL_SOURCE_DIR}/lib/cmake/cub/"
  )
endmacro()

macro(cccl_get_cudatoolkit)
  find_package(CUDAToolkit REQUIRED)
endmacro()

macro(cccl_get_cudax)
  find_package(
    cudax
    CONFIG
    REQUIRED
    NO_DEFAULT_PATH # Only check the explicit HINTS below:
    HINTS "${CCCL_SOURCE_DIR}/lib/cmake/cudax/"
  )
endmacro()

macro(cccl_get_dlpack)
  include("${_cccl_cpm_file}")
  CPMAddPackage("gh:dmlc/dlpack#v1.2")
endmacro()

macro(cccl_get_libcudacxx)
  find_package(
    libcudacxx
    CONFIG
    REQUIRED
    NO_DEFAULT_PATH # Only check the explicit HINTS below:
    HINTS "${CCCL_SOURCE_DIR}/lib/cmake/libcudacxx/"
  )
endmacro()

set(
  CCCL_NVBENCH_SHA
  "56d552687e6a462a812d6f046f5a85a07f13c9f3"
  CACHE STRING
  "SHA/tag to use for CCCL's NVBench."
)
mark_as_advanced(CCCL_NVBENCH_SHA)
macro(cccl_get_nvbench)
  include("${_cccl_cpm_file}")
  CPMAddPackage("gh:NVIDIA/nvbench#${CCCL_NVBENCH_SHA}")
endmacro()

# CCCL-specific NVBench utilities
macro(cccl_get_nvbench_helper)
  if (NOT TARGET cccl.nvbench_helper)
    add_subdirectory(
      "${CCCL_SOURCE_DIR}/nvbench_helper"
      "${CCCL_BINARY_DIR}/nvbench_helper"
    )
  endif()
endmacro()

macro(cccl_get_nvtx)
  include("${_cccl_cpm_file}")
  CPMAddPackage(
    NAME NVTX
    GITHUB_REPOSITORY NVIDIA/NVTX
    GIT_TAG release-v3
    DOWNLOAD_ONLY ON
    SYSTEM ON
  )
  include("${NVTX_SOURCE_DIR}/c/nvtxImportedTargets.cmake")
endmacro()

set(
  CCCL_RAPIDS_CMAKE_SHA
  "6d7c911330acc35ec547c42943e1fc8f4b21c27a"
  CACHE STRING
  "SHA/tag to use for CCCL's rapids-cmake test utilities."
)
mark_as_advanced(CCCL_RAPIDS_CMAKE_SHA)
# Download the rapids-cmake CTest GPU resource helpers. Nothing from the
# package is configured or built; only the included CMake files are used.
macro(cccl_get_rapids_test)
  include("${_cccl_cpm_file}")
  CPMAddPackage(
    NAME rapids-cmake
    GITHUB_REPOSITORY rapidsai/rapids-cmake
    GIT_TAG ${CCCL_RAPIDS_CMAKE_SHA}
    DOWNLOAD_ONLY ON
  )
  include("${rapids-cmake_SOURCE_DIR}/rapids-cmake/test/init.cmake")
  include("${rapids-cmake_SOURCE_DIR}/rapids-cmake/test/gpu_requirements.cmake")
endmacro()

macro(cccl_get_thrust)
  find_package(
    Thrust
    CONFIG
    REQUIRED
    NO_DEFAULT_PATH # Only check the explicit HINTS below:
    HINTS "${CCCL_SOURCE_DIR}/lib/cmake/thrust/"
  )
endmacro()

macro(cccl_get_nccl)
  list(APPEND CMAKE_MODULE_PATH "${_cccl_find_module_dir}")
  find_package(NCCL ${ARGN})
  list(POP_BACK CMAKE_MODULE_PATH)
endmacro()
