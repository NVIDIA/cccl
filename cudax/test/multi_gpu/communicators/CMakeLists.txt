#===----------------------------------------------------------------------===##
#
# Part of CUDA Experimental in CUDA C++ Core Libraries,
# under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#
#===----------------------------------------------------------------------===##

set(multi_gpu_communicator_test_sources nccl_communicator.cu)

function(cudax_add_multi_gpu_communicator_test target_name_var source)
  cmake_path(GET source STEM filename)
  set(test_target "cudax.test.multi_gpu.communicators.${filename}")

  cccl_add_executable(${test_target} SOURCES "${source}" ADD_CTEST)
  target_link_libraries(${test_target} PRIVATE cudax.compiler_interface)

  set(${target_name_var} ${test_target} PARENT_SCOPE)
endfunction()

foreach (source IN LISTS multi_gpu_communicator_test_sources)
  cudax_add_multi_gpu_communicator_test(test_target "${source}")
endforeach()
