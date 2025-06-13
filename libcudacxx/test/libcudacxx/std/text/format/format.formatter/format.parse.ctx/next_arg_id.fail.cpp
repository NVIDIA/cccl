//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr size_t next_arg_id()

#include <cuda/std/__format_>

__host__ __device__ constexpr bool test()
{
  // [format.parse.ctx]/8
  // Let cur-arg-id be the value of next_arg_id_ prior to this call. Call
  // expressions where cur-arg-id >= num_args_ is true are not core constant
  // expressions (7.7 [expr.const]).
  cuda::std::format_parse_context context("", 0);
  context.next_arg_id();

  return true;
}

__host__ __device__ void f()
{
#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  static_assert(test());
#else // ^^^ _CCCL_BUILTIN_IS_CONSTANT_EVALUATED ^^^ / vvv !_CCCL_BUILTIN_IS_CONSTANT_EVALUATED vvv
  static_assert(false);
#endif // ^^^ !_CCCL_BUILTIN_IS_CONSTANT_EVALUATED ^^^
}
