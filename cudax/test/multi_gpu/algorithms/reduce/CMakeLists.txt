#===----------------------------------------------------------------------===##
#
# Part of CUDA Experimental in CUDA C++ Core Libraries,
# under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#
#===----------------------------------------------------------------------===##

if (NOT cudax_ENABLE_NCCL)
  return()
endif()

file(GLOB test_srcs LIST_DIRECTORIES FALSE CONFIGURE_DEPENDS *.cu *.cpp)

foreach (src IN LISTS test_srcs)
  cudax_add_multi_gpu_test("algorithms.reduce" test_target "${src}")
  target_link_libraries(${test_target} PRIVATE NCCL::nccl)
endforeach()
