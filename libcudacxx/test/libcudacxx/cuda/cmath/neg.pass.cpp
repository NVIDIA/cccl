//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/cmath>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

template <class T>
__host__ __device__ constexpr void test_neg(T pos, T neg)
{
  assert(cuda::neg(pos) == neg);
  assert(cuda::neg(neg) == pos);
}

template <class T>
__host__ __device__ constexpr void test_type()
{
  static_assert(cuda::std::is_same_v<decltype(cuda::neg(T{})), T>);
  static_assert(noexcept(cuda::neg(T{})));

  test_neg<T>(0, 0);
  test_neg<T>(1, T(-1));
  test_neg<T>(4, T(-4));
  test_neg<T>(29, T(-29));
  test_neg<T>(127, T(-127));
  test_neg<T>(cuda::std::numeric_limits<T>::max(), cuda::std::numeric_limits<T>::min() + 1);
  test_neg<T>(cuda::std::numeric_limits<T>::min(), cuda::std::numeric_limits<T>::min());
}

__host__ __device__ constexpr bool test()
{
  test_type<signed char>();
  test_type<signed short>();
  test_type<signed int>();
  test_type<signed long>();
  test_type<signed long long>();
#if _CCCL_HAS_INT128()
  test_type<__int128_t>();
#endif // _CCCL_HAS_INT128()

  test_type<unsigned char>();
  test_type<unsigned short>();
  test_type<unsigned int>();
  test_type<unsigned long>();
  test_type<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_type<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
