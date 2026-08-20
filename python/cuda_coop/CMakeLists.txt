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
  set(CCCL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
  set(_cuda_coop_include_dir "cuda/coop/_headers/include")
  foreach (_cuda_coop_header_root cub/cub thrust/thrust libcudacxx/include/cuda)
    if (NOT EXISTS "${CCCL_SOURCE_DIR}/${_cuda_coop_header_root}")
      message(
        FATAL_ERROR
        "CUDA_COOP_INSTALL_HEADER_BUNDLE=ON requires a full CCCL source tree; "
        "missing ${CCCL_SOURCE_DIR}/${_cuda_coop_header_root}."
      )
    endif()
  endforeach()

  install(
    DIRECTORY "${CCCL_SOURCE_DIR}/cub/cub"
    DESTINATION "${_cuda_coop_include_dir}"
    FILES_MATCHING
    PATTERN "*.cuh"
  )
  install(
    DIRECTORY "${CCCL_SOURCE_DIR}/thrust/thrust"
    DESTINATION "${_cuda_coop_include_dir}"
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.inl"
  )
  install(
    DIRECTORY "${CCCL_SOURCE_DIR}/libcudacxx/include/cuda"
    DESTINATION "${_cuda_coop_include_dir}"
    PATTERN "CMakeLists.txt" EXCLUDE
  )
  if (EXISTS "${CCCL_SOURCE_DIR}/libcudacxx/include/nv")
    install(
      DIRECTORY "${CCCL_SOURCE_DIR}/libcudacxx/include/nv"
      DESTINATION "${_cuda_coop_include_dir}"
      PATTERN "CMakeLists.txt" EXCLUDE
    )
  endif()

  if (CUDA_COOP_CCCL_SOURCE_REVISION)
    set(_cuda_coop_revision "${CUDA_COOP_CCCL_SOURCE_REVISION}")
  else()
    execute_process(
      COMMAND git -C "${CCCL_SOURCE_DIR}" rev-parse HEAD
      RESULT_VARIABLE _cuda_coop_revision_result
      OUTPUT_VARIABLE _cuda_coop_revision
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (NOT _cuda_coop_revision_result EQUAL 0)
      set(_cuda_coop_revision "unknown")
      message(
        STATUS
        "Git metadata is unavailable; recording an unknown CCCL source revision"
      )
    endif()
  endif()
  if (NOT _cuda_coop_revision MATCHES "^[0-9A-Za-z._+-]+$")
    message(
      FATAL_ERROR
      "CUDA_COOP_CCCL_SOURCE_REVISION must be a revision token; got "
      "${_cuda_coop_revision}"
    )
  endif()
  set(
    _cuda_coop_bundle_provenance
    "${CMAKE_CURRENT_BINARY_DIR}/cccl-bundle-provenance.json"
  )
  file(
    WRITE "${_cuda_coop_bundle_provenance}"
    "{\n  \"cccl_source_commit\": \"${_cuda_coop_revision}\"\n}\n"
  )
  install(
    FILES "${_cuda_coop_bundle_provenance}"
    DESTINATION cuda/coop/_headers
  )
endif()
