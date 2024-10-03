set(_cccl_cpm_file "${CMAKE_CURRENT_LIST_DIR}/CPM.cmake")

macro(cccl_get_boost)
  include("${_cccl_cpm_file}")
  CPMAddPackage("gh:boostorg/boost#boost-1.83.0")
endmacro()

macro(cccl_get_catch2)
  include("${_cccl_cpm_file}")
  CPMAddPackage("gh:catchorg/Catch2@2.13.9")
endmacro()

macro(cccl_get_fmt)
  include("${_cccl_cpm_file}")
  CPMAddPackage("gh:fmtlib/fmt#11.0.1")
endmacro()

macro(cccl_get_nvbench)
  include("${_cccl_cpm_file}")
  CPMAddPackage("gh:NVIDIA/nvbench#main")
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
