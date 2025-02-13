//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// template<class T>
//   complex<T>
//   operator/(const T& lhs, const complex<T>& rhs);

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "test_macros.h"

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  T lhs(-8.5);
  cuda::std::complex<T> rhs(1.5, 2.5);
  cuda::std::complex<T> x(-1.5, 2.5);
  assert(lhs / rhs == x);

  return true;
}

int main(int, char**)
{
  test<float>();
  test<double>();
#ifdef _LIBCUDACXX_HAS_NVFP16
  test<__half>();
#endif
#ifdef _LIBCUDACXX_HAS_NVBF16
  test<__nv_bfloat16>();
#endif
// CUDA treats long double as double
//  test<long double>();
#if !defined(_LIBCUDACXX_HAS_NO_CONSTEXPR_COMPLEX_OPERATIONS)
  static_assert(test<float>(), "");
  static_assert(test<double>(), "");
// CUDA treats long double as double
//  static_assert(test<long double>(), "");
#endif // !_LIBCUDACXX_HAS_NO_CONSTEXPR_COMPLEX_OPERATIONS

  return 0;
}
