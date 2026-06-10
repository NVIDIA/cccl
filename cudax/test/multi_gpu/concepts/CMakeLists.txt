#===----------------------------------------------------------------------===##
#
# Part of CUDA Experimental in CUDA C++ Core Libraries,
# under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#
#===----------------------------------------------------------------------===##

set(
  multi_gpu_concepts_test_sources
  communicator.cu
  has_all_gather.cu
  has_all_reduce.cu
  has_all_to_all.cu
  has_all_to_all_v.cu
  has_broadcast.cu
  has_gather.cu
  has_gather_v.cu
  has_recv.cu
  has_reduce.cu
  has_send.cu
  has_synchronous_all_gather.cu
  has_synchronous_all_reduce.cu
  has_synchronous_all_to_all.cu
  has_synchronous_all_to_all_v.cu
  has_synchronous_broadcast.cu
  has_synchronous_gather.cu
  has_synchronous_gather_v.cu
  has_synchronous_recv.cu
  has_synchronous_reduce.cu
  has_synchronous_send.cu
  synchronous_communicator.cu
)

function(cudax_add_multi_gpu_concepts_test target_name_var source)
  cmake_path(GET source STEM filename)
  set(test_target "cudax.test.multi_gpu.concepts.${filename}")

  cccl_add_executable(${test_target} SOURCES "${source}" ADD_CTEST)
  target_link_libraries(${test_target} PRIVATE cudax.compiler_interface)

  set(${target_name_var} ${test_target} PARENT_SCOPE)
endfunction()

foreach (source IN LISTS multi_gpu_concepts_test_sources)
  cudax_add_multi_gpu_concepts_test(test_target "${source}")
endforeach()
