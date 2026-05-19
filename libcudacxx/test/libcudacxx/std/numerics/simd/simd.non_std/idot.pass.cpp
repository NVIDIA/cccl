//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/__simd/idot.h>

// template<class T, class U, class Abi, class AccT>
//   constexpr AccT cuda::simd::idot(
//     const basic_vec<T, Abi>& lhs, const basic_vec<U, Abi>& rhs, AccT acc) noexcept;

#include <cuda/simd>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

namespace simd = cuda::std::simd;

template <typename T, int N>
using fixed_size_vec = simd::basic_vec<T, simd::fixed_size<N>>;

template <typename LhsVec, typename RhsVec, typename AccT, typename = void>
inline constexpr bool has_idot = false;

template <typename LhsVec, typename RhsVec, typename AccT>
inline constexpr bool
  has_idot<LhsVec,
           RhsVec,
           AccT,
           cuda::std::void_t<decltype(cuda::simd::idot(
             cuda::std::declval<LhsVec>(), cuda::std::declval<RhsVec>(), cuda::std::declval<AccT>()))>> = true;

template <typename T, typename U, typename AccT, int N>
TEST_FUNC constexpr AccT
scalar_idot(const cuda::std::array<T, N>& lhs_values, const cuda::std::array<U, N>& rhs_values, AccT acc)
{
  AccT result = acc;
  for (int i = 0; i < N; ++i)
  {
    AccT lhs_value = static_cast<AccT>(lhs_values[i]);
    AccT rhs_value = static_cast<AccT>(rhs_values[i]);
    AccT product   = static_cast<AccT>(lhs_value * rhs_value);
    result         = static_cast<AccT>(result + product);
  }
  return result;
}

template <typename T, typename U, typename AccT, int N>
TEST_FUNC constexpr void test_values(cuda::std::array<T, N> lhs_values, cuda::std::array<U, N> rhs_values, AccT acc)
{
  using LhsVec = simd::basic_vec<T, simd::fixed_size<N>>;
  using RhsVec = simd::basic_vec<U, simd::fixed_size<N>>;
  LhsVec lhs(lhs_values);
  RhsVec rhs(rhs_values);

  static_assert(cuda::std::is_same_v<decltype(cuda::simd::idot(lhs, rhs, acc)), AccT>);
  static_assert(noexcept(cuda::simd::idot(lhs, rhs, acc)));

  AccT result   = cuda::simd::idot(lhs, rhs, acc);
  AccT expected = scalar_idot<T, U, AccT, N>(lhs_values, rhs_values, acc);
  assert(result == expected);
}

template <typename T, typename U, typename AccT, int N>
TEST_FUNC constexpr void test_generated(AccT acc)
{
  cuda::std::array<T, N> lhs_values{};
  cuda::std::array<U, N> rhs_values{};
  for (int i = 0; i < N; ++i)
  {
    if constexpr (cuda::std::is_signed_v<T>)
    {
      lhs_values[i] = static_cast<T>((i % 5) - 2);
    }
    else
    {
      lhs_values[i] = static_cast<T>((i % 5) + 1);
    }

    if constexpr (cuda::std::is_signed_v<U>)
    {
      rhs_values[i] = static_cast<U>((i % 7) - 3);
    }
    else
    {
      rhs_values[i] = static_cast<U>((i % 7) + 2);
    }
  }
  test_values<T, U, AccT, N>(lhs_values, rhs_values, acc);
}

TEST_FUNC constexpr void test_8bit_dp4a()
{
  {
    cuda::std::array<int8_t, 4> lhs_values{-8, -3, 2, 7};
    cuda::std::array<int8_t, 4> rhs_values{4, -5, 6, -7};
    test_values<int8_t, int8_t, int32_t, 4>(lhs_values, rhs_values, int32_t{11});
  }
  {
    cuda::std::array<uint8_t, 7> lhs_values{1, 2, 3, 4, 5, 6, 7};
    cuda::std::array<uint8_t, 7> rhs_values{8, 7, 6, 5, 4, 3, 2};
    test_values<uint8_t, uint8_t, uint32_t, 7>(lhs_values, rhs_values, uint32_t{13});
  }
  {
    cuda::std::array<uint8_t, 5> lhs_values{1, 2, 3, 4, 5};
    cuda::std::array<int8_t, 5> rhs_values{-1, 2, -3, 4, -5};
    test_values<uint8_t, int8_t, int32_t, 5>(lhs_values, rhs_values, int32_t{-17});
  }
  {
    cuda::std::array<int8_t, 3> lhs_values{-4, 5, -6};
    cuda::std::array<uint8_t, 3> rhs_values{7, 8, 9};
    test_values<int8_t, uint8_t, int32_t, 3>(lhs_values, rhs_values, int32_t{19});
  }
}

TEST_FUNC constexpr void test_16bit_8bit_dp2a()
{
  {
    cuda::std::array<int16_t, 5> lhs_values{-300, 20, 45, -12, 17};
    cuda::std::array<int8_t, 5> rhs_values{3, -4, 5, -6, 7};
    test_values<int16_t, int8_t, int32_t, 5>(lhs_values, rhs_values, int32_t{23});
  }
  {
    cuda::std::array<int8_t, 5> lhs_values{3, -4, 5, -6, 7};
    cuda::std::array<int16_t, 5> rhs_values{-300, 20, 45, -12, 17};
    test_values<int8_t, int16_t, int32_t, 5>(lhs_values, rhs_values, int32_t{29});
  }
  {
    cuda::std::array<uint16_t, 5> lhs_values{300, 20, 45, 12, 17};
    cuda::std::array<uint8_t, 5> rhs_values{3, 4, 5, 6, 7};
    test_values<uint16_t, uint8_t, uint32_t, 5>(lhs_values, rhs_values, uint32_t{31});
  }
  {
    cuda::std::array<uint8_t, 5> lhs_values{3, 4, 5, 6, 7};
    cuda::std::array<uint16_t, 5> rhs_values{300, 20, 45, 12, 17};
    test_values<uint8_t, uint16_t, uint32_t, 5>(lhs_values, rhs_values, uint32_t{37});
  }
  {
    cuda::std::array<int16_t, 5> lhs_values{-300, 20, 45, -12, 17};
    cuda::std::array<uint8_t, 5> rhs_values{3, 200, 5, 255, 7};
    test_values<int16_t, uint8_t, int32_t, 5>(lhs_values, rhs_values, int32_t{41});
  }
  {
    cuda::std::array<uint8_t, 5> lhs_values{3, 200, 5, 255, 7};
    cuda::std::array<int16_t, 5> rhs_values{-300, 20, 45, -12, 17};
    test_values<uint8_t, int16_t, int32_t, 5>(lhs_values, rhs_values, int32_t{-47});
  }
  {
    cuda::std::array<uint16_t, 5> lhs_values{300, 40000, 45, 65535, 17};
    cuda::std::array<int8_t, 5> rhs_values{3, -4, 5, -6, 7};
    test_values<uint16_t, int8_t, int32_t, 5>(lhs_values, rhs_values, int32_t{43});
  }
  {
    cuda::std::array<int8_t, 5> lhs_values{3, -4, 5, -6, 7};
    cuda::std::array<uint16_t, 5> rhs_values{300, 40000, 45, 65535, 17};
    test_values<int8_t, uint16_t, int32_t, 5>(lhs_values, rhs_values, int32_t{-53});
  }
}

TEST_FUNC constexpr bool test_all()
{
  static_assert(!has_idot<fixed_size_vec<float, 4>, fixed_size_vec<float, 4>, int>);
  static_assert(!has_idot<fixed_size_vec<int, 4>, fixed_size_vec<float, 4>, int>);
  static_assert(!has_idot<fixed_size_vec<int, 4>, fixed_size_vec<int, 4>, float>);

  test_8bit_dp4a();
  test_16bit_8bit_dp2a();

  test_generated<short, short, int, 3>(5);
  test_generated<int, unsigned, long long, 5>(-7);
  test_generated<long, long long, long long, 4>(9);
  test_generated<unsigned short, unsigned, unsigned long long, 6>(11);
  test_generated<unsigned long, unsigned long long, unsigned long long, 3>(13);
#if _CCCL_HAS_INT128()
  test_generated<__int128_t, __uint128_t, __int128_t, 3>(__int128_t{17});
  test_generated<__uint128_t, __uint128_t, __uint128_t, 5>(__uint128_t{19});
#endif // _CCCL_HAS_INT128()

  return true;
}

int main(int, char**)
{
  assert(test_all());
  static_assert(test_all());
  return 0;
}
