//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/functional>
#include <cuda/std/cassert>

#include "test_macros.h"

template <typename OpT, typename T>
__host__ __device__ constexpr void test_op(const T lhs, const T rhs, const T expected)
{
  assert((OpT{}(lhs, rhs) == expected) && (OpT{}(lhs, rhs) == OpT{}(rhs, lhs)));
}

template <typename T>
__host__ __device__ constexpr void test(T lhs, T rhs, T expected)
{
  test_op<cuda::minimum<T>>(lhs, rhs, expected);
  test_op<cuda::minimum<>>(lhs, rhs, expected);
  test_op<cuda::minimum<void>>(lhs, rhs, expected);
}

__host__ __device__ constexpr bool test()
{
  test<int>(0, 1, 0);
  test<int>(1, 0, 0);
  test<int>(0, 0, 0);
  test<int>(-1, 1, -1);
  test<char>('a', 'b', 'a');
  test<float>(1.0f, 2.0f, 1.0f);
  test<double>(1.0f, 2.0f, 1.0f);
#if _CCCL_HAS_FLOAT128()
  test<__float128>(__float128(1.0f), __float128(2.0f), __float128(1.0f));
#endif
  return true;
}

__host__ __device__ bool runtime_test()
{
#if _CCCL_CTK_AT_LEAST(12, 2) // < CTK 12.2 does not support == operator for __half and __nv_bfloat16
#  if _CCCL_HAS_NVFP16()
  test<__half>(__half(1.0f), __half(2.0f), __half(1.0f));
#  endif
#  if _CCCL_HAS_NVBF16()
  test<__nv_bfloat16>(__nv_bfloat16(1.0f), __nv_bfloat16(2.0f), __nv_bfloat16(1.0f));
#  endif
#endif // _CCCL_CTK_AT_LEAST(12, 2)
  return true;
}

int main(int, char**)
{
  assert(test());
  assert(runtime_test());
  static_assert(test());
  return 0;
}
