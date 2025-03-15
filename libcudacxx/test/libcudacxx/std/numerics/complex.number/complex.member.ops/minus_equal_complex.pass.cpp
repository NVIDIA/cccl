//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// complex& operator-=(const complex& rhs);

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "test_macros.h"

template <class T>
__host__ __device__ constexpr bool test()
{
  cuda::std::complex<T> c;
  const cuda::std::complex<T> c2(1.5, 2.5);
  assert(c.real() == T(0));
  assert(c.imag() == T(0));
  c -= c2;
  assert(c.real() == T(-1.5));
  assert(c.imag() == T(-2.5));
  c -= c2;
  assert(c.real() == T(-3));
  assert(c.imag() == T(-5));

  cuda::std::complex<T> c3;

  c3 = c;
  cuda::std::complex<int> ic(1, 1);
  c3 -= ic;
  assert(c3.real() == T(-4));
  assert(c3.imag() == T(-6));

  c3 = c;
  cuda::std::complex<float> fc(1, 1);
  c3 -= fc;
  assert(c3.real() == T(-4));
  assert(c3.imag() == T(-6));

  return true;
}

int main(int, char**)
{
  test<float>();
  test<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  test<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()

  static_assert(test<float>(), "");
  static_assert(test<double>(), "");
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert(test<long double>(), "");
#endif // _CCCL_HAS_LONG_DOUBLE()

  return 0;
}
