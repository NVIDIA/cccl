#===----------------------------------------------------------------------===##
#
# Part of CUDA Experimental in CUDA C++ Core Libraries,
# under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#
#===----------------------------------------------------------------------===##

function(cudax_add_multi_gpu_test sub_prefix target_name_var source)
  cmake_path(GET source STEM filename)
  cudax_add_catch2_test("${target_name_var}" "multi_gpu.${sub_prefix}.${filename}" "${source}" ${ARGN})
  target_include_directories(
    "${${target_name_var}}"
    PRIVATE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/include"
  )
  set("${target_name_var}" "${${target_name_var}}" PARENT_SCOPE)
endfunction()

add_subdirectory(algorithms)
add_subdirectory(communicators)
add_subdirectory(concepts)
add_subdirectory(nccl)
