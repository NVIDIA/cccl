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
//   bool
//   operator==(const T& lhs, const complex<T>& rhs);

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "test_macros.h"

template <class T>
__host__ __device__ constexpr void test_constexpr()
{
  {
    constexpr T lhs(-2.5);
    constexpr cuda::std::complex<T> rhs(1.5, 2.5);
    static_assert(!(lhs == rhs), "");
  }
  {
    constexpr T lhs(-2.5);
    constexpr cuda::std::complex<T> rhs(1.5, 0);
    static_assert(!(lhs == rhs), "");
  }
  {
    constexpr T lhs(1.5);
    constexpr cuda::std::complex<T> rhs(1.5, 2.5);
    static_assert(!(lhs == rhs), "");
  }
  {
    constexpr T lhs(1.5);
    constexpr cuda::std::complex<T> rhs(1.5, 0);
    static_assert(lhs == rhs, "");
  }
}

template <class T>
__host__ __device__ constexpr void test_nonconstexpr()
{
  {
    T lhs(-2.5);
    cuda::std::complex<T> rhs(1.5, 2.5);
    assert(!(lhs == rhs));
  }
  {
    T lhs(-2.5);
    cuda::std::complex<T> rhs(1.5, 0);
    assert(!(lhs == rhs));
  }
  {
    T lhs(1.5);
    cuda::std::complex<T> rhs(1.5, 2.5);
    assert(!(lhs == rhs));
  }
  {
    T lhs(1.5);
    cuda::std::complex<T> rhs(1.5, 0);
    assert(lhs == rhs);
  }
}

template <class T>
__host__ __device__ constexpr bool test()
{
  test_nonconstexpr<T>();
  test_constexpr<T>();

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
  test_nonconstexpr<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test_nonconstexpr<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()
  //     test_constexpr<int>();
  static_assert(test<float>(), "");
  static_assert(test<double>(), "");
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert(test<long double>(), "");
#endif // _CCCL_HAS_LONG_DOUBLE()

  return 0;
}
