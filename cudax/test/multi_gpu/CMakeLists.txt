#===----------------------------------------------------------------------===##
#
# Part of CUDA Experimental in CUDA C++ Core Libraries,
# under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#
#===----------------------------------------------------------------------===##

cccl_get_nccl(QUIET)

option(
  cudax_ENABLE_NCCL
  "Enable cudax tests that link against NCCL."
  "${NCCL_FOUND}"
)

# TODO(jfaibussowit):
#
# Remove this once NCCL supports windows and we have added NCCL to the NVHPC
# devcontainers.
macro(hack_cudax_handle_missing_nccl)
  if (NOT cudax_ENABLE_NCCL)
    message(
      AUTHOR_WARNING
      "Function should only be called if we have requested NCCL, this is a developer error"
    )
  endif()

  option(
    HACK_cudax_ALLOW_MISSING_NCCL
    "Hack to work around CI issues since NCCL does not (yet) officially support Windows"
    TRUE
  )

  if (NCCL_FOUND)
    # If we have somehow already found NCCL (e.g. they have shipped support but we have
    # forgotten to remove this function), then we have nothing to do. We still call
    # cccl_get_nccl() though because we want the found version to appear in the logs. The
    # previous call was with QUIET which won't do that.
    cccl_get_nccl(REQUIRED)
  elseif (HACK_cudax_ALLOW_MISSING_NCCL)
    set(cudax_ENABLE_NCCL FALSE)
    set(cudax_ENABLE_NCCL FALSE CACHE BOOL "" FORCE)
  else()
    # This will cause a hard error
    cccl_get_nccl(REQUIRED)
  endif()
endmacro()

if (cudax_ENABLE_NCCL)
  if (WIN32 OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "NVHPC"))
    hack_cudax_handle_missing_nccl()
  else()
    cccl_get_nccl(REQUIRED)
  endif()
endif()

function(cudax_add_multi_gpu_test sub_prefix target_name_var source)
  cmake_path(GET source STEM filename)
  cudax_add_catch2_test("${target_name_var}" "multi_gpu.${sub_prefix}.${filename}" ${source} ${ARGN})
  set(${target_name_var} ${test_target} PARENT_SCOPE)
endfunction()

add_subdirectory(algorithms)
add_subdirectory(communicators)
add_subdirectory(concepts)
add_subdirectory(nccl)
