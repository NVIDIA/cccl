//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/cmath>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_rsqrt()
{
  constexpr bool is_integer = cuda::std::is_integral_v<T>;
  using R                   = cuda::std::conditional_t<is_integer, double, T>;
  static_assert(cuda::std::is_same_v<R, decltype(cuda::rsqrt(T{}))>);
  static_assert(noexcept(cuda::rsqrt(T{})));

  assert(cuda::rsqrt(T{0}) == cuda::std::numeric_limits<R>::infinity());
  if constexpr (!is_integer)
  {
    assert(cuda::rsqrt(-T{0} == -cuda::std::numeric_limits<R>::infinity()));
  }
  assert(cuda::rsqrt(T{1} == R{1.0}));
  assert(cuda::rsqrt(T{4}) == R{0.5});
  assert(cuda::rsqrt(T{16}) == R{0.25});
  if constexpr (cuda::std::is_signed_v<T>)
  {
    assert(cuda::std::isnan(cuda::rsqrt(T{-1})));
    assert(cuda::std::isnan(cuda::rsqrt(T{-4})));
    assert(cuda::std::isnan(cuda::rsqrt(T{-16})));
  }
  if constexpr (!is_integer)
  {
    assert(cuda::rsqrt(cuda::std::numeric_limits<T>::infinity()) == R{0.0});
    assert(cuda::std::isnan(cuda::rsqrt(-cuda::std::numeric_limits<T>::infinity())));
    assert(cuda::std::isnan(cuda::rsqrt(cuda::std::numeric_limits<T>::quiet_NaN())));
    assert(cuda::std::isnan(cuda::rsqrt(-cuda::std::numeric_limits<T>::quiet_NaN())));
  }
}

__host__ __device__ bool test()
{
  test_rsqrt<float>();
  test_rsqrt<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_rsqrt<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
  test_rsqrt<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_rsqrt<__nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16()

  test_rsqrt<signed char>();
  test_rsqrt<signed short>();
  test_rsqrt<signed int>();
  test_rsqrt<signed long>();
  test_rsqrt<signed long long>();
#if _CCCL_HAS_INT128()
  test_rsqrt<__int128_t>();
#endif // _CCCL_HAS_INT128()

  test_rsqrt<unsigned char>();
  test_rsqrt<unsigned short>();
  test_rsqrt<unsigned int>();
  test_rsqrt<unsigned long>();
  test_rsqrt<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_rsqrt<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  return true;
}

int main(int, char**)
{
  test();
  return 0;
}
