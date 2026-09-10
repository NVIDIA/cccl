//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/simd>

// template<class T, class U, class Abi, class AccT = common_type_t<T, U>>
//   constexpr AccT cuda::simd::dot(
//     const basic_vec<T, Abi>& lhs, const basic_vec<U, Abi>& rhs, AccT acc = {}) noexcept;

#include <cuda/simd>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

namespace simd = cuda::std::simd;

template <typename T, int N>
using fixed_size_vec = simd::basic_vec<T, simd::fixed_size<N>>;

template <typename LhsVec, typename RhsVec, typename AccT, typename = void>
inline constexpr bool has_dot = false;

template <typename LhsVec, typename RhsVec, typename AccT>
inline constexpr bool
  has_dot<LhsVec,
          RhsVec,
          AccT,
          cuda::std::void_t<decltype(cuda::simd::dot(
            cuda::std::declval<LhsVec>(), cuda::std::declval<RhsVec>(), cuda::std::declval<AccT>()))>> = true;

template <typename T, typename U, typename AccT, int N>
TEST_FUNC constexpr AccT
scalar_dot(const cuda::std::array<T, N>& lhs_values, const cuda::std::array<U, N>& rhs_values, AccT acc)
{
  auto result = acc;
  for (int i = 0; i < N; ++i)
  {
    auto lhs_value = static_cast<AccT>(lhs_values[i]);
    auto rhs_value = static_cast<AccT>(rhs_values[i]);
    auto product   = static_cast<AccT>(lhs_value * rhs_value);
    result         = static_cast<AccT>(result + product);
  }
  return result;
}

template <typename T, typename U, typename AccT, int N>
TEST_FUNC constexpr void test_values(cuda::std::array<T, N> lhs_values, cuda::std::array<U, N> rhs_values, AccT acc)
{
  using LhsVec      = simd::basic_vec<T, simd::fixed_size<N>>;
  using RhsVec      = simd::basic_vec<U, simd::fixed_size<N>>;
  using DefaultAccT = cuda::std::common_type_t<T, U>;
  LhsVec lhs(lhs_values, simd::flag_convert);
  RhsVec rhs(rhs_values, simd::flag_convert);

  static_assert(cuda::std::is_same_v<decltype(cuda::simd::dot(lhs, rhs, acc)), AccT>);
  static_assert(noexcept(cuda::simd::dot(lhs, rhs, acc)));
  static_assert(cuda::std::is_same_v<decltype(cuda::simd::dot(lhs, rhs)), DefaultAccT>);
  static_assert(noexcept(cuda::simd::dot(lhs, rhs)));

  AccT result                  = cuda::simd::dot(lhs, rhs, acc);
  AccT expected                = scalar_dot<T, U, AccT, N>(lhs_values, rhs_values, acc);
  DefaultAccT default_result   = cuda::simd::dot(lhs, rhs);
  DefaultAccT default_expected = scalar_dot<T, U, DefaultAccT, N>(lhs_values, rhs_values, DefaultAccT{});
  assert(result == expected);
  assert(default_result == default_expected);
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
  { // 8-bit x 8-bit (signed)
    cuda::std::array<int8_t, 4> lhs_values{-8, -3, 2, 7};
    cuda::std::array<int8_t, 4> rhs_values{4, -5, 6, -7};
    test_values<int8_t, int8_t, int, 4>(lhs_values, rhs_values, 11);
  }
  { // 8-bit x 8-bit (unsigned)
    cuda::std::array<uint8_t, 7> lhs_values{1, 2, 3, 128, 5, 200, 255};
    cuda::std::array<uint8_t, 7> rhs_values{255, 7, 128, 5, 200, 3, 2};
    test_values<uint8_t, uint8_t, unsigned, 7>(lhs_values, rhs_values, 13);
  }
  { // 8-bit x 8-bit (unsigned x signed)
    cuda::std::array<uint8_t, 5> lhs_values{1, 2, 3, 4, 5};
    cuda::std::array<int8_t, 5> rhs_values{-1, 2, -3, 4, -5};
    test_values<uint8_t, int8_t, int, 5>(lhs_values, rhs_values, -17);
  }
  { // 8-bit x 8-bit (signed x unsigned)
    cuda::std::array<int8_t, 3> lhs_values{-4, 5, -6};
    cuda::std::array<uint8_t, 3> rhs_values{7, 8, 9};
    test_values<int8_t, uint8_t, int, 3>(lhs_values, rhs_values, 19);
  }
  { // 8-bit x 8-bit (unsigned)
    cuda::std::array<uint8_t, 4> lhs_values{1, 1, 1, 1};
    cuda::std::array<uint8_t, 4> rhs_values{1, 1, 1, 1};
    test_values<uint8_t, uint8_t, unsigned, 4>(lhs_values, rhs_values, 0xFFFFFFFD);
  }
}

TEST_FUNC constexpr void test_16bit_8bit_dp2a()
{
  { // 16-bit x 8-bit (signed)
    cuda::std::array<int16_t, 5> lhs_values{-300, 20, 45, -12, 17};
    cuda::std::array<int8_t, 5> rhs_values{3, -4, 5, -6, 7};
    test_values<int16_t, int8_t, int, 5>(lhs_values, rhs_values, 23);
  }
  { // 8-bit x 16-bit
    cuda::std::array<int8_t, 5> lhs_values{3, -4, 5, -6, 7};
    cuda::std::array<int16_t, 5> rhs_values{-300, 20, 45, -12, 17};
    test_values<int8_t, int16_t, int, 5>(lhs_values, rhs_values, 29);
  }
  { // 16-bit x 8-bit (unsigned)
    cuda::std::array<uint16_t, 5> lhs_values{300, 20, 45, 12, 17};
    cuda::std::array<uint8_t, 5> rhs_values{3, 4, 5, 6, 7};
    test_values<uint16_t, uint8_t, unsigned, 5>(lhs_values, rhs_values, 31);
  }
  { // 8-bit x 16-bit (unsigned)
    cuda::std::array<uint8_t, 5> lhs_values{3, 4, 5, 6, 7};
    cuda::std::array<uint16_t, 5> rhs_values{300, 20, 45, 12, 17};
    test_values<uint8_t, uint16_t, unsigned, 5>(lhs_values, rhs_values, 37);
  }
  { // 16-bit x 8-bit (signed x unsigned)
    cuda::std::array<int16_t, 5> lhs_values{-300, 20, 45, -12, 17};
    cuda::std::array<uint8_t, 5> rhs_values{3, 200, 5, 255, 7};
    test_values<int16_t, uint8_t, int, 5>(lhs_values, rhs_values, 41);
  }
  { // 8-bit x 16-bit (unsigned x signed)
    cuda::std::array<uint8_t, 5> lhs_values{3, 200, 5, 255, 7};
    cuda::std::array<int16_t, 5> rhs_values{-300, 20, 45, -12, 17};
    test_values<uint8_t, int16_t, int, 5>(lhs_values, rhs_values, -47);
  }
  { // 16-bit x 8-bit (unsigned x signed)
    cuda::std::array<uint16_t, 5> lhs_values{300, 40000, 45, 65535, 17};
    cuda::std::array<int8_t, 5> rhs_values{3, -4, 5, -6, 7};
    test_values<uint16_t, int8_t, int, 5>(lhs_values, rhs_values, 43);
  }
  { // 8-bit x 16-bit (signed x unsigned)
    cuda::std::array<int8_t, 5> lhs_values{3, -4, 5, -6, 7};
    cuda::std::array<uint16_t, 5> rhs_values{300, 40000, 45, 65535, 17};
    test_values<int8_t, uint16_t, int, 5>(lhs_values, rhs_values, -53);
  }
}

TEST_FUNC constexpr void test_non_integer()
{
  {
    cuda::std::array<float, 4> lhs_values{1.5f, -2.0f, 0.25f, 4.0f};
    cuda::std::array<double, 4> rhs_values{2.0, 3.0, -4.0, 0.5};
    test_values<float, double, double, 4>(lhs_values, rhs_values, 0.25);
  }
  {
    cuda::std::array<int8_t, 4> lhs_values{1, -2, 3, -4};
    cuda::std::array<uint8_t, 4> rhs_values{5, 6, 7, 8};
    test_values<int8_t, uint8_t, float, 4>(lhs_values, rhs_values, 0.5f);
  }
}

TEST_FUNC void test_complex()
{
  using complex = cuda::std::complex<float>;
  cuda::std::array<complex, 3> lhs_values{complex{1.0f, 2.0f}, complex{-3.0f, 0.5f}, complex{2.0f, -1.0f}};
  cuda::std::array<complex, 3> rhs_values{complex{0.5f, -1.0f}, complex{2.0f, 3.0f}, complex{-1.0f, 4.0f}};
  test_values<complex, complex, complex, 3>(lhs_values, rhs_values, complex{1.0f, -2.0f});
}

TEST_FUNC constexpr bool test_all()
{
  static_assert(has_dot<fixed_size_vec<float, 4>, fixed_size_vec<float, 4>, int>);
  static_assert(has_dot<fixed_size_vec<int, 4>, fixed_size_vec<float, 4>, int>);
  static_assert(has_dot<fixed_size_vec<int, 4>, fixed_size_vec<int, 4>, float>);

  test_8bit_dp4a();
  test_16bit_8bit_dp2a();
  test_non_integer();

  // test mixed types
  test_generated<short, short, int, 3>(5);
  test_generated<int, unsigned, long long, 5>(-7);
  test_generated<long, long long, long long, 4>(9);
  test_generated<unsigned short, unsigned, unsigned long long, 6>(11);
  test_generated<unsigned long, unsigned long long, unsigned long long, 3>(13);
  test_generated<int8_t, int8_t, unsigned, 5>(17);
  test_generated<uint16_t, uint16_t, unsigned, 5>(19);
#if _CCCL_HAS_INT128()
  test_generated<__int128_t, __uint128_t, __int128_t, 3>(__int128_t{17});
  test_generated<__uint128_t, __uint128_t, __uint128_t, 5>(__uint128_t{19});
#endif // _CCCL_HAS_INT128()

  return true;
}

int main(int, char**)
{
  assert(test_all());
  test_complex();
  static_assert(test_all());
  return 0;
}
