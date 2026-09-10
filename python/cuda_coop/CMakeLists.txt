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
option(
  CUDA_COOP_ALLOW_DIRTY_HEADER_BUNDLE
  "Allow locally modified CCCL headers with unknown bundle provenance"
  OFF
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

  # Git searches parent directories for repository metadata. Require the CCCL
  # source directory itself to be the worktree root so an unpacked archive
  # nested in an unrelated checkout cannot inherit that checkout's revision.
  set(_cuda_coop_source_is_worktree_root OFF)
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
    set(_cuda_coop_source_is_worktree_root ON)
  endif()

  set(_cuda_coop_header_bundle_unverified OFF)
  if (_cuda_coop_source_is_worktree_root)
    # Match the headers and CMake package sources selected by the CCCL install
    # rules, the rules and package configuration controlling the bundle, and the
    # additional public CUDAX *.h headers installed below. Include ignored files
    # because CMake's directory install rules do not consult Git ignore rules.
    set(
      _cuda_coop_bundle_input_pathspecs
      ":(glob)cub/cub/**/*.cuh"
      ":(glob)thrust/thrust/**/*.h"
      ":(glob)thrust/thrust/**/*.inl"
      ":(glob)cudax/include/cuda/**/*.cuh"
      ":(glob)cudax/include/cuda/**/*.h"
      ":(glob)libcudacxx/include/cuda/**"
      ":(glob)libcudacxx/include/nv/**"
      ":(exclude,glob)libcudacxx/include/cuda/**/CMakeLists.txt"
      ":(exclude,glob)libcudacxx/include/nv/**/CMakeLists.txt"
      ":(glob)lib/cmake/cccl/**"
      ":(glob)lib/cmake/cub/**"
      ":(glob)lib/cmake/cudax/**"
      ":(glob)lib/cmake/libcudacxx/**"
      ":(glob)lib/cmake/thrust/**"
      "cmake/CCCLInstallRules.cmake"
      "cmake/install/cccl.cmake"
      "cmake/install/cub.cmake"
      "cmake/install/cudax.cmake"
      "cmake/install/libcudacxx.cmake"
      "cmake/install/thrust.cmake"
      "python/cuda_coop/CMakeLists.txt"
    )
    execute_process(
      COMMAND
        git -C "${CCCL_SOURCE_DIR}" status --porcelain=v1 --untracked-files=all
        --ignored=matching -- ${_cuda_coop_bundle_input_pathspecs}
      RESULT_VARIABLE _cuda_coop_header_status_result
      OUTPUT_VARIABLE _cuda_coop_header_status
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (NOT "${_cuda_coop_header_status_result}" STREQUAL "0")
      set(_cuda_coop_header_bundle_unverified ON)
      set(
        _cuda_coop_header_bundle_diagnostic
        "Git could not verify the CCCL header bundle inputs"
      )
    elseif (NOT "${_cuda_coop_header_status}" STREQUAL "")
      set(_cuda_coop_header_bundle_unverified ON)
      set(
        _cuda_coop_header_bundle_diagnostic
        "The CCCL header bundle inputs contain local changes"
      )
    endif()
  endif()

  if (_cuda_coop_header_bundle_unverified)
    if (NOT CUDA_COOP_ALLOW_DIRTY_HEADER_BUNDLE)
      message(
        FATAL_ERROR
        "${_cuda_coop_header_bundle_diagnostic}. Commit or remove those changes, "
        "or explicitly configure CUDA_COOP_ALLOW_DIRTY_HEADER_BUNDLE=ON to "
        "build with unknown provenance."
      )
    endif()
    set(_cuda_coop_source_revision "unknown")
    message(
      WARNING
      "${_cuda_coop_header_bundle_diagnostic}; recording an unknown CCCL "
      "source revision"
    )
  elseif (CUDA_COOP_CCCL_SOURCE_REVISION)
    set(_cuda_coop_source_revision "${CUDA_COOP_CCCL_SOURCE_REVISION}")
  elseif (_cuda_coop_source_is_worktree_root)
    execute_process(
      COMMAND git -C "${CCCL_SOURCE_DIR}" rev-parse HEAD
      RESULT_VARIABLE _cuda_coop_source_revision_result
      OUTPUT_VARIABLE _cuda_coop_source_revision
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  else()
    set(_cuda_coop_source_revision_result 1)
  endif()
  if (
    DEFINED _cuda_coop_source_revision_result
    AND NOT "${_cuda_coop_source_revision_result}" STREQUAL "0"
  )
    set(_cuda_coop_source_revision "unknown")
    message(
      WARNING
      "Git metadata is unavailable; recording an unknown CCCL source revision"
    )
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
