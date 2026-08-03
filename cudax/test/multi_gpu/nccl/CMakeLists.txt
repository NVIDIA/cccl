#===----------------------------------------------------------------------===##
#
# Part of CUDA Experimental in CUDA C++ Core Libraries,
# under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#
#===----------------------------------------------------------------------===##

file(GLOB test_srcs LIST_DIRECTORIES FALSE CONFIGURE_DEPENDS *.cu *.cpp)

# These tests do not require NCCL. The ones that exercise the NCCL API guard themselves with
# _CCCL_HAS_NCCL(), so only link NCCL when it is available.
foreach (src IN LISTS test_srcs)
  cudax_add_multi_gpu_test("nccl" test_target "${src}")
  if (cudax_ENABLE_NCCL)
    target_link_libraries(${test_target} PRIVATE NCCL::nccl)
  endif()
endforeach()
