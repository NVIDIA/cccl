//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11, c++14

#include <cuda/__floating_point_>
#include <cuda/std/cassert>

#include "test_macros.h"

template <class Lhs, class Rhs, class Exp>
__host__ __device__ constexpr void test_mul(Lhs lhs, Rhs rhs, Exp expected)
{
  static_assert(noexcept(lhs * rhs));
  ASSERT_SAME_TYPE(decltype(lhs * rhs), Exp);
  assert(lhs * rhs == expected);
}

__host__ __device__ constexpr bool test()
{
  test_mul(cuda::fp16{1.0}, cuda::fp16{2.0}, cuda::fp16{2.0});
  test_mul(cuda::fp16{1.0}, cuda::fp32{2.0}, cuda::fp32{2.0});
  test_mul(cuda::fp16{1.0}, cuda::fp64{2.0}, cuda::fp64{2.0});
  test_mul(cuda::fp16{1.0}, float{2.0}, float{2.0});
  test_mul(cuda::fp16{1.0}, double{2.0}, double{2.0});
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  test_mul(cuda::fp16{1.0}, long double{2.0}, long double{2.0});
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

  test_mul(cuda::fp32{1.0}, cuda::fp16{2.0}, cuda::fp32{2.0});
  test_mul(cuda::fp32{1.0}, cuda::fp32{2.0}, cuda::fp32{2.0});
  test_mul(cuda::fp32{1.0}, cuda::fp64{2.0}, cuda::fp64{2.0});
  // test_mul(cuda::fp32{1.0}, cuda::bf16{2.0}, cuda::fp32{2.0});
  test_mul(cuda::fp32{1.0}, float{2.0}, cuda::fp32{2.0});
  test_mul(cuda::fp32{1.0}, double{2.0}, double{2.0});
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  test_mul(cuda::fp32{1.0}, long double{2.0}, long double{2.0});
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

  test_mul(cuda::fp64{1.0}, cuda::fp16{2.0}, cuda::fp64{2.0});
  test_mul(cuda::fp64{1.0}, cuda::fp32{2.0}, cuda::fp64{2.0});
  test_mul(cuda::fp64{1.0}, cuda::fp64{2.0}, cuda::fp64{2.0});
  // test_mul(cuda::fp64{1.0}, cuda::bf16{2.0}, cuda::fp64{2.0});
  test_mul(cuda::fp64{1.0}, float{2.0}, cuda::fp64{2.0});
  test_mul(cuda::fp64{1.0}, double{2.0}, cuda::fp64{2.0});
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  test_mul(cuda::fp64{1.0}, long double{2.0}, long double{2.0});
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

  // test_mul(cuda::bf16{1.0}, cuda::fp32{2.0}, cuda::fp32{2.0});
  // test_mul(cuda::bf16{1.0}, cuda::fp64{2.0}, cuda::fp64{2.0});
  // test_mul(cuda::bf16{1.0}, cuda::bf16{2.0}, cuda::bf16{2.0});
  // test_mul(cuda::bf16{1.0}, float{2.0}, float{2.0});
  // test_mul(cuda::bf16{1.0}, double{2.0}, double{2.0});
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  // test_mul(cuda::bf16{1.0}, long double{2.0}, long double{2.0});
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

  return true;
}

int main(int, char**)
{
  test();
  // static_assert(test());

  return 0;
}
