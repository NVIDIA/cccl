##===----------------------------------------------------------------------===##
##
## Part of nvrtcc in CUDA Core Compute Libraries,
## under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
##
##===----------------------------------------------------------------------===##

# 1. Add --diag-suppress test.
add_compile_only_test(test.nvcc_options.diag_suppress diag_suppress.cu)
target_compile_options(
  test.nvcc_options.diag_suppress
  PRIVATE "-Werror=all-warnings" "--diag-suppress=186"
)

add_compile_only_test(test.nvcc_options.diag_suppress.short diag_suppress.cu)
target_compile_options(
  test.nvcc_options.diag_suppress.short
  PRIVATE "-Werror=all-warnings" "-diag-suppress=186"
)
