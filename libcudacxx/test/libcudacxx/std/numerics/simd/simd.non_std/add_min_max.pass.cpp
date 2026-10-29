//===----------------------------------------------------------------------===//
//
// Part of libcu++, the CUDA C++ Standard Library,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/simd>

// template<class T, class Abi>
//   constexpr basic_vec<T, Abi> cuda::simd::add_max(
//     const basic_vec<T, Abi>& a, const basic_vec<T, Abi>& b, const basic_vec<T, Abi>& c) noexcept;
//
// template<class T, class Abi>
//   constexpr basic_vec<T, Abi> cuda::simd::add_min(
//     const basic_vec<T, Abi>& a, const basic_vec<T, Abi>& b, const basic_vec<T, Abi>& c) noexcept;
//
// template<class T, class Abi>
//   constexpr basic_vec<T, Abi> cuda::simd::add_max_relu(
//     const basic_vec<T, Abi>& a, const basic_vec<T, Abi>& b, const basic_vec<T, Abi>& c) noexcept;
//
// template<class T, class Abi>
//   constexpr basic_vec<T, Abi> cuda::simd::add_min_relu(
//     const basic_vec<T, Abi>& a, const basic_vec<T, Abi>& b, const basic_vec<T, Abi>& c) noexcept;

#include <cuda/simd>
#include <cuda/std/algorithm>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

namespace simd = cuda::std::simd;

template <typename T, int N>
using fixed_size_vec = simd::basic_vec<T, simd::fixed_size<N>>;

template <typename Vec, typename = void>
inline constexpr bool has_add_min_max = false;

template <typename Vec>
inline constexpr bool has_add_min_max<
  Vec,
  cuda::std::void_t<
    decltype(cuda::simd::add_max(cuda::std::declval<Vec>(), cuda::std::declval<Vec>(), cuda::std::declval<Vec>())),
    decltype(cuda::simd::add_min(cuda::std::declval<Vec>(), cuda::std::declval<Vec>(), cuda::std::declval<Vec>()))>> =
  true;

template <typename Vec, typename = void>
inline constexpr bool has_add_min_max_relu = false;

template <typename Vec>
inline constexpr bool has_add_min_max_relu<
  Vec,
  cuda::std::void_t<
    decltype(cuda::simd::add_max_relu(cuda::std::declval<Vec>(), cuda::std::declval<Vec>(), cuda::std::declval<Vec>())),
    decltype(cuda::simd::add_min_relu(cuda::std::declval<Vec>(), cuda::std::declval<Vec>(), cuda::std::declval<Vec>()))>> =
  true;

template <typename T>
TEST_FUNC constexpr T scalar_add_max(T a, T b, T c)
{
  T sum = static_cast<T>(a + b);
  return cuda::std::max(sum, c);
}

template <typename T>
TEST_FUNC constexpr T scalar_add_min(T a, T b, T c)
{
  T sum = static_cast<T>(a + b);
  return cuda::std::min(sum, c);
}

template <typename T>
TEST_FUNC constexpr T scalar_add_max_relu(T a, T b, T c)
{
  return cuda::std::max(scalar_add_max(a, b, c), T{0});
}

template <typename T>
TEST_FUNC constexpr T scalar_add_min_relu(T a, T b, T c)
{
  return cuda::std::max(scalar_add_min(a, b, c), T{0});
}

template <typename T, int N>
TEST_FUNC constexpr void
test_values(cuda::std::array<T, N> a_values, cuda::std::array<T, N> b_values, cuda::std::array<T, N> c_values)
{
  using vec_t = fixed_size_vec<T, N>;
  vec_t a(a_values);
  vec_t b(b_values);
  vec_t c(c_values);

  static_assert(cuda::std::is_same_v<decltype(cuda::simd::add_max(a, b, c)), vec_t>);
  static_assert(cuda::std::is_same_v<decltype(cuda::simd::add_min(a, b, c)), vec_t>);
  static_assert(noexcept(cuda::simd::add_max(a, b, c)));
  static_assert(noexcept(cuda::simd::add_min(a, b, c)));

  vec_t maximum = cuda::simd::add_max(a, b, c);
  vec_t minimum = cuda::simd::add_min(a, b, c);
  for (int i = 0; i < N; ++i)
  {
    assert(maximum[i] == scalar_add_max(a_values[i], b_values[i], c_values[i]));
    assert(minimum[i] == scalar_add_min(a_values[i], b_values[i], c_values[i]));
  }

  if constexpr (cuda::std::is_signed_v<T>)
  {
    static_assert(cuda::std::is_same_v<decltype(cuda::simd::add_max_relu(a, b, c)), vec_t>);
    static_assert(cuda::std::is_same_v<decltype(cuda::simd::add_min_relu(a, b, c)), vec_t>);
    static_assert(noexcept(cuda::simd::add_max_relu(a, b, c)));
    static_assert(noexcept(cuda::simd::add_min_relu(a, b, c)));

    vec_t maximum_relu = cuda::simd::add_max_relu(a, b, c);
    vec_t minimum_relu = cuda::simd::add_min_relu(a, b, c);
    for (int i = 0; i < N; ++i)
    {
      assert(maximum_relu[i] == scalar_add_max_relu(a_values[i], b_values[i], c_values[i]));
      assert(minimum_relu[i] == scalar_add_min_relu(a_values[i], b_values[i], c_values[i]));
    }
  }
}

// small values that mix signs (for signed types) and cover N < 3 and large sizes
template <typename T, int N>
TEST_FUNC constexpr void test_generated_size()
{
  cuda::std::array<T, N> a_values{};
  cuda::std::array<T, N> b_values{};
  cuda::std::array<T, N> c_values{};
  constexpr int offset = cuda::std::is_signed_v<T> ? 3 : 0;
  for (int i = 0; i < N; ++i)
  {
    a_values[i] = static_cast<T>(i % 7 - offset);
    b_values[i] = static_cast<T>(i % 5 - offset / 2);
    c_values[i] = static_cast<T>(i % 3 - offset / 3);
  }
  test_values<T, N>(a_values, b_values, c_values);
}

template <typename T, int N>
TEST_FUNC constexpr void test_size()
{
  constexpr auto min_val = cuda::std::numeric_limits<T>::min();
  constexpr auto max_val = cuda::std::numeric_limits<T>::max();

  if constexpr (cuda::std::is_signed_v<T>)
  {
    cuda::std::array<T, N> a_values{max_val, min_val, T{10}};
    cuda::std::array<T, N> b_values{T{0}, T{0}, T{-20}};
    cuda::std::array<T, N> c_values{static_cast<T>(max_val - T{1}), static_cast<T>(min_val + T{1}), T{-5}};
    if constexpr (N > 3)
    {
      a_values[3] = T{2};
      b_values[3] = T{3};
      c_values[3] = T{5};
    }
    if constexpr (N > 4)
    {
      a_values[4] = T{-2};
      b_values[4] = T{2};
      c_values[4] = T{0};
    }
    test_values<T, N>(a_values, b_values, c_values);
  }
  else
  {
    cuda::std::array<T, N> a_values{max_val, min_val, T{10}};
    cuda::std::array<T, N> b_values{T{1}, T{1}, T{20}};
    cuda::std::array<T, N> c_values{T{5}, T{2}, T{15}};
    if constexpr (N > 3)
    {
      a_values[3] = T{2};
      b_values[3] = T{3};
      c_values[3] = T{5};
    }
    if constexpr (N > 4)
    {
      a_values[4] = T{0};
      b_values[4] = T{0};
      c_values[4] = T{0};
    }
    test_values<T, N>(a_values, b_values, c_values);
  }
}

template <typename T>
TEST_FUNC constexpr void test()
{
  test_size<T, 3>();
  test_size<T, 4>();
  test_size<T, 5>();
  test_generated_size<T, 1>();
  test_generated_size<T, 2>();
  test_generated_size<T, 7>();
  test_generated_size<T, 32>();
}

TEST_FUNC constexpr bool test_all()
{
  static_assert(!has_add_min_max<fixed_size_vec<float, 4>>);
  static_assert(has_add_min_max_relu<fixed_size_vec<int, 4>>);
  static_assert(!has_add_min_max_relu<fixed_size_vec<unsigned, 4>>);
  static_assert(!has_add_min_max_relu<fixed_size_vec<float, 4>>);

  test<signed char>();
  test<signed short>();
  test<signed int>();
  test<signed long>();
  test<signed long long>();
#if _CCCL_HAS_INT128()
  test<__int128_t>();
#endif // _CCCL_HAS_INT128()

  test<unsigned char>();
  test<unsigned short>();
  test<unsigned int>();
  test<unsigned long>();
  test<unsigned long long>();
#if _CCCL_HAS_INT128()
  test<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  return true;
}

int main(int, char**)
{
  assert(test_all());
  static_assert(test_all());

  return 0;
}
