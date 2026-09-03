# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21...3.31 FATAL_ERROR)

project(cuda_coop DESCRIPTION "Python package cuda-coop" LANGUAGES NONE)

option(
  CUDA_COOP_INSTALL_HEADER_BUNDLE
  "Install cuda-coop's private CCCL header bundle"
  ON
)
set(
  CUDA_COOP_CCCL_SOURCE_REVISION
  ""
  CACHE STRING
  "CCCL source revision recorded in the cuda-coop header bundle"
)
if (CUDA_COOP_INSTALL_HEADER_BUNDLE)
  # cuda-coop owns its bundled CCCL headers. Follow the repository's install
  # rules so CUB, Thrust, libcu++, and CUDAX stay on one source revision.
  set(CCCL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
  set(
    _cuda_coop_cccl_install_rules
    "${CCCL_SOURCE_DIR}/cmake/CCCLInstallRules.cmake"
  )
  if (NOT EXISTS "${_cuda_coop_cccl_install_rules}")
    message(
      FATAL_ERROR
      "CUDA_COOP_INSTALL_HEADER_BUNDLE=ON requires a full CCCL source tree, "
      "but CCCL install rules were not found at "
      "${_cuda_coop_cccl_install_rules}."
    )
  endif()

  set(CCCL_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  set(CCCL_TOPLEVEL_PROJECT OFF)
  set(CCCL_ENABLE_CUDAX ON)
  set(CCCL_ENABLE_INSTALL_RULES ON)
  set(libcudacxx_ENABLE_INSTALL_RULES ON)
  set(CUB_ENABLE_INSTALL_RULES ON)
  set(Thrust_ENABLE_INSTALL_RULES ON)
  set(cudax_ENABLE_INSTALL_RULES ON)
  set(CMAKE_INSTALL_INCLUDEDIR "cuda/coop/_headers/include" CACHE PATH "" FORCE)
  set(CMAKE_INSTALL_LIBDIR "cuda/coop/_headers/lib" CACHE PATH "" FORCE)
  include("${_cuda_coop_cccl_install_rules}")

  if (CUDA_COOP_CCCL_SOURCE_REVISION)
    set(_cuda_coop_source_revision "${CUDA_COOP_CCCL_SOURCE_REVISION}")
  else()
    # Git searches parent directories for repository metadata. Require the CCCL
    # source directory itself to be the worktree root so an unpacked archive
    # nested in an unrelated checkout cannot inherit that checkout's revision.
    set(_cuda_coop_source_revision_result 1)
    execute_process(
      COMMAND git -C "${CCCL_SOURCE_DIR}" rev-parse --show-prefix
      RESULT_VARIABLE _cuda_coop_source_prefix_result
      OUTPUT_VARIABLE _cuda_coop_source_prefix
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (
      "${_cuda_coop_source_prefix_result}" STREQUAL "0"
      AND "${_cuda_coop_source_prefix}" STREQUAL ""
    )
      execute_process(
        COMMAND git -C "${CCCL_SOURCE_DIR}" rev-parse HEAD
        RESULT_VARIABLE _cuda_coop_source_revision_result
        OUTPUT_VARIABLE _cuda_coop_source_revision
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )
    endif()
    if (NOT "${_cuda_coop_source_revision_result}" STREQUAL "0")
      set(_cuda_coop_source_revision "unknown")
      message(
        WARNING
        "Git metadata is unavailable; recording an unknown CCCL source revision"
      )
    endif()
  endif()
  if (NOT _cuda_coop_source_revision MATCHES "^[0-9A-Za-z._+-]+$")
    message(
      FATAL_ERROR
      "CUDA_COOP_CCCL_SOURCE_REVISION must be a revision token; got "
      "${_cuda_coop_source_revision}"
    )
  endif()
  set(
    _cuda_coop_bundle_provenance
    "${CMAKE_CURRENT_BINARY_DIR}/cccl-bundle-provenance.json"
  )
  file(
    WRITE "${_cuda_coop_bundle_provenance}"
    "{\n  \"cccl_source_commit\": \"${_cuda_coop_source_revision}\"\n}\n"
  )
  install(
    FILES "${_cuda_coop_bundle_provenance}"
    DESTINATION cuda/coop/_headers
  )

  # CUDAX's standard install rule selects public *.cuh headers. Its public
  # entry points also include *.h implementation headers, so stage those too.
  install(
    DIRECTORY "${CCCL_SOURCE_DIR}/cudax/include/cuda"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
    PATTERN "*.h"
  )
endif()
