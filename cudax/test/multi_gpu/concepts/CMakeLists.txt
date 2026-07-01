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

foreach (src IN LISTS test_srcs)
  cudax_add_multi_gpu_test("concepts" test_target "${src}")
endforeach()
