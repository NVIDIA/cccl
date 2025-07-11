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

template <class T, class U>
__host__ __device__ constexpr void test_uabs(T input, U ref)
{
  assert(cuda::uabs(input) == ref);
}

template <class T>
__host__ __device__ constexpr void test_type()
{
  using U = cuda::std::make_unsigned_t<T>;

  static_assert(cuda::std::is_same_v<decltype(cuda::uabs(T{})), U>);
  static_assert(noexcept(cuda::uabs(T{})));

  test_uabs(T(0), U(0));
  test_uabs(T(1), U(1));
  test_uabs(T(100), U(100));
  if constexpr (cuda::std::is_signed_v<T>)
  {
    test_uabs(T(-1), U(1));
    test_uabs(T(-100), U(100));
    test_uabs(T(cuda::std::numeric_limits<T>::min() + 1), static_cast<U>(cuda::std::numeric_limits<T>::max()));
    test_uabs(cuda::std::numeric_limits<T>::min(), U(static_cast<U>(cuda::std::numeric_limits<T>::max()) + 1));
  }
  test_uabs(cuda::std::numeric_limits<T>::max(), static_cast<U>(cuda::std::numeric_limits<T>::max()));
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
