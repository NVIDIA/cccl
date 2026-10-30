#===----------------------------------------------------------------------===##
#
# Part of CUDA Experimental in CUDA C++ Core Libraries,
# under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#
#===----------------------------------------------------------------------===##

file(
  GLOB_RECURSE test_srcs
  LIST_DIRECTORIES FALSE
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  CONFIGURE_DEPENDS
  "*.cu"
)

foreach (src IN LISTS test_srcs)
  string(REGEX REPLACE "\\.cu$" "" test_name "${src}")
  string(REPLACE "/" "." test_name "${test_name}")
  string(PREPEND test_name "coop.")
  cudax_add_catch2_test("test_target" "${test_name}" "${src}")
endforeach()
