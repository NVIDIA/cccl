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

template <class U, class T>
__host__ __device__ constexpr void test_unegate(U input, T ref)
{
  assert(cuda::__unegate(input) == ref);
}

template <class T>
__host__ __device__ constexpr void test_type()
{
  using U = cuda::std::make_unsigned_t<T>;

  static_assert(cuda::std::is_same_v<decltype(cuda::__unegate(U{})), T>);
  static_assert(noexcept(cuda::uabs(T{})));

  test_unegate(U(0), T(0));
  test_unegate(U(1), T(-1));
  test_unegate(U(100), T(-100));
  test_unegate(U(cuda::std::numeric_limits<T>::max()), T(cuda::std::numeric_limits<T>::min() + 1));
  test_unegate(U(U(cuda::std::numeric_limits<T>::max()) + 1), cuda::std::numeric_limits<T>::min());
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

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
