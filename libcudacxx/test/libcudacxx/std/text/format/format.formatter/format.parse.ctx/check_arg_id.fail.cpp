//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr void check_arg_id(size_t id);

#include <cuda/std/__format_>

__host__ __device__ constexpr bool test()
{
  // [format.parse.ctx]/11
  // Remarks: Call expressions where id >= num_args_ are not
  // core constant expressions ([expr.const]).
  cuda::std::format_parse_context context("", 0);
  context.check_arg_id(1);

  return true;
}

void f()
{
#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  static_assert(test());
#else // ^^^ _CCCL_BUILTIN_IS_CONSTANT_EVALUATED ^^^ / vvv !_CCCL_BUILTIN_IS_CONSTANT_EVALUATED vvv
  static_assert(false);
#endif // ^^^ !_CCCL_BUILTIN_IS_CONSTANT_EVALUATED ^^^
}
