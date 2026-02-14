//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/utility>

// template<typename _Tp, typename _Up>
//   constexpr bool in_range(_Tp v, _Up start, _Up end) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/type_traits>
#include <cuda/utility>

#include "test_macros.h"

template <typename T>
__host__ __device__ constexpr void test()
{
  static_assert(noexcept(cuda::in_range(cuda::std::declval<T>(), cuda::std::declval<T>(), cuda::std::declval<T>())));
  assert(cuda::in_range(T{5}, T{0}, T{10}));
  assert(!cuda::in_range(T{15}, T{0}, T{10}));
  assert(cuda::in_range(T{10}, T{0}, T{10})); // test bound
  assert(cuda::in_range(T{10}, T{10}, T{10})); // test bound
  if constexpr (cuda::std::is_signed_v<T>)
  {
    assert(!cuda::in_range(T{-5}, T{0}, T{10}));
    assert(cuda::in_range(T{5}, T{-10}, T{10}));
    assert(cuda::in_range(T{5}, T{-1}, T{10}));
    assert(cuda::in_range(T{0}, T{-1}, T{1}));
  }
  assert(!cuda::in_range(T{5}, T{cuda::std::numeric_limits<T>::max() - T{1}}, cuda::std::numeric_limits<T>::max()));
  assert(!cuda::in_range(T{5}, cuda::std::numeric_limits<T>::min(), T{cuda::std::numeric_limits<T>::min() + T{1}}));
  assert(cuda::in_range(T{5}, cuda::std::numeric_limits<T>::min(), cuda::std::numeric_limits<T>::max()));

  if constexpr (cuda::std::numeric_limits<T>::has_infinity)
  {
    constexpr auto inf = cuda::std::numeric_limits<T>::infinity();
    assert(cuda::in_range(inf, -inf, inf));
    assert(cuda::in_range(-inf, -inf, inf));
    assert(!cuda::in_range(inf, T{0}, T{10}));
    assert(cuda::in_range(T{1}, T{-1}, inf));
  }
  if constexpr (cuda::std::numeric_limits<T>::has_quiet_NaN)
  {
    constexpr auto nan = cuda::std::numeric_limits<T>::quiet_NaN();
    if (!cuda::std::__cccl_default_is_constant_evaluated())
    {
      assert(!cuda::in_range(nan, T{0}, T{10}));
      assert(!cuda::in_range(T{1}, nan, T{10}));
      assert(!cuda::in_range(T{1}, T{10}, nan));
    }
  }
  if constexpr (cuda::std::numeric_limits<T>::has_quiet_NaN && cuda::std::numeric_limits<T>::has_infinity)
  {
    constexpr auto inf = cuda::std::numeric_limits<T>::infinity();
    constexpr auto nan = cuda::std::numeric_limits<T>::quiet_NaN();
    if (!cuda::std::__cccl_default_is_constant_evaluated())
    {
      assert(!cuda::in_range(nan, -inf, inf));
      assert(!cuda::in_range(T{1}, nan, inf));
      assert(!cuda::in_range(T{1}, inf, nan));
    }
  }
}

__host__ __device__ constexpr bool test()
{
  test<unsigned char>();
  test<signed char>();
  test<unsigned short>();
  test<short>();
  test<unsigned int>();
  test<int>();
  test<unsigned long>();
  test<long>();
  test<unsigned long long>();
  test<long long>();
#if _CCCL_HAS_INT128()
  test<__int128_t>();
  test<__uint128_t>();
#endif // _CCCL_HAS_INT128()
  test<float>();
  test<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
  return true;
}

__host__ __device__ bool runtime_test()
{
#if _CCCL_HAS_NVFP16() && _CCCL_CTK_AT_LEAST(12, 2)
  test<__half>();
#endif
#if _CCCL_HAS_NVBF16() && _CCCL_CTK_AT_LEAST(12, 2)
  test<__nv_bfloat16>();
#endif
  return true;
}

int main(int, char**)
{
  static_assert(test());
  assert(test());
  assert(runtime_test());
  return 0;
}
