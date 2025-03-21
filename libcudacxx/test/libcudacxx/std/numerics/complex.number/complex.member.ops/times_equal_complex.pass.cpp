//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// complex& operator*=(const complex& rhs);

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "test_macros.h"

TEST_DIAG_SUPPRESS_MSVC(4244) // conversion from 'const double' to 'int', possible loss of data

template <class T>
__host__ __device__ constexpr bool test()
{
  cuda::std::complex<T> c(1);
  const cuda::std::complex<T> c2(1.5, 2.5);
  assert(c.real() == T(1));
  assert(c.imag() == T(0));
  c *= c2;
  assert(c.real() == T(1.5));
  assert(c.imag() == T(2.5));
  c *= c2;
  assert(c.real() == T(-4));
  assert(c.imag() == T(7.5));

  cuda::std::complex<T> c3;

  c3 = c;
  cuda::std::complex<int> ic(1, 1);
  c3 *= ic;
  assert(c3.real() == T(-11.5));
  assert(c3.imag() == T(3.5));

  c3 = c;
  cuda::std::complex<float> fc(1, 1);
  c3 *= fc;
  assert(c3.real() == T(-11.5));
  assert(c3.imag() == T(3.5));

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
#if _LIBCUDACXX_HAS_CONSTEXPR_COMPLEX_OPERATIONS()
#  if !TEST_COMPILER(GCC, <, 8) // GCC 7 does not support constexpr is_nan and friends
  static_assert(test<float>(), "");
  static_assert(test<double>(), "");
#    if _CCCL_HAS_LONG_DOUBLE()
  static_assert(test<long double>(), "");
#    endif // _CCCL_HAS_LONG_DOUBLE()
#  endif // !TEST_COMPILER(GCC, <, 8)
#endif // _LIBCUDACXX_HAS_CONSTEXPR_COMPLEX_OPERATIONS()

  return 0;
}
