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
//   conj(const complex<T>& x);

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "test_macros.h"

template <class T>
__host__ __device__ constexpr void test(const cuda::std::complex<T>& z, cuda::std::complex<T> x)
{
  assert(conj(z) == x);
}

template <class T>
__host__ __device__ constexpr bool test()
{
  test(cuda::std::complex<T>(1, 2), cuda::std::complex<T>(1, -2));
  test(cuda::std::complex<T>(-1, 2), cuda::std::complex<T>(-1, -2));
  test(cuda::std::complex<T>(1, -2), cuda::std::complex<T>(1, 2));
  test(cuda::std::complex<T>(-1, -2), cuda::std::complex<T>(-1, 2));

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
