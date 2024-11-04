//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _TEST_LIBCUDACXX_CUDA_FUNCTIONAL_MIN_MAX_COMMON_H
#define _TEST_LIBCUDACXX_CUDA_FUNCTIONAL_MIN_MAX_COMMON_H

#include <cuda/std/cassert>

namespace
{

template <typename OpT, typename T, T lhs, T rhs, T expected>
__host__ __device__ constexpr bool test_op()
{
  constexpr auto op = OpT{};

  assert(op(lhs, rhs) == expected);

  if (rhs == lhs)
  {
    assert(op(lhs, rhs) == op(rhs, lhs));
  }
  else
  {
    assert(op(lhs, rhs) != op(rhs, lhs));
  }
  return true;
}

} // namespace

#endif // _TEST_LIBCUDACXX_CUDA_FUNCTIONAL_MIN_MAX_COMMON_H
