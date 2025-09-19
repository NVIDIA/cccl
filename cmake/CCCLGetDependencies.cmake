set(_cccl_cpm_file "${CMAKE_CURRENT_LIST_DIR}/CPM.cmake")

macro(cccl_get_boost)
  include("${_cccl_cpm_file}")
  CPMAddPackage("gh:boostorg/boost#boost-1.83.0")
endmacro()

# The CCCL Catch2Helper library:
macro(cccl_get_c2h)
  if (NOT TARGET cccl.c2h)
    add_subdirectory("${CCCL_SOURCE_DIR}/c2h" "${CCCL_BINARY_DIR}/c2h")
  endif()
endmacro()

macro(cccl_get_catch2)
  include("${_cccl_cpm_file}")
  CPMAddPackage("gh:catchorg/Catch2@3.8.0")
endmacro()

macro(cccl_get_fmt)
  include("${_cccl_cpm_file}")
  CPMAddPackage("gh:fmtlib/fmt#11.0.1")
endmacro()

macro(cccl_get_json)
  include("${_cccl_cpm_file}")
  CPMAddPackage("gh:nlohmann/json@3.12.0")
endmacro()

set(CCCL_NVBENCH_SHA "0c24f0250bf4414ab5ad19709090c6396e76516b" CACHE STRING "SHA/tag to use for CCCL's NVBench.")
mark_as_advanced(CCCL_NVBENCH_SHA)
macro(cccl_get_nvbench)
  include("${_cccl_cpm_file}")
  CPMAddPackage("gh:NVIDIA/nvbench#${CCCL_NVBENCH_SHA}")
endmacro()

# CCCL-specific NVBench utilities
macro(cccl_get_nvbench_helper)
  if (NOT TARGET cccl.nvbench_helper)
    add_subdirectory("${CCCL_SOURCE_DIR}/nvbench_helper" "${CCCL_BINARY_DIR}/nvbench_helper")
  endif()
endmacro()

macro(cccl_get_nvtx)
  include("${_cccl_cpm_file}")
  CPMAddPackage(
    NAME NVTX
    GITHUB_REPOSITORY NVIDIA/NVTX
    GIT_TAG release-v3
    DOWNLOAD_ONLY
    SYSTEM
  )
  include("${NVTX_SOURCE_DIR}/c/nvtxImportedTargets.cmake")
endmacro()
