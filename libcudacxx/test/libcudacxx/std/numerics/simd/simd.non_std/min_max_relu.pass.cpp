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

// template<class T, class Abi>
//   constexpr basic_vec<T, Abi> cuda::simd::max_relu(
//     const basic_vec<T, Abi>& lhs, const basic_vec<T, Abi>& rhs) noexcept;
//
// template<class T, class Abi>
//   constexpr basic_vec<T, Abi> cuda::simd::min_relu(
//     const basic_vec<T, Abi>& lhs, const basic_vec<T, Abi>& rhs) noexcept;
//
// template<class T, class Abi>
//   constexpr basic_vec<T, Abi> cuda::simd::max_relu(
//     const basic_vec<T, Abi>& a, const basic_vec<T, Abi>& b, const basic_vec<T, Abi>& c) noexcept;
//
// template<class T, class Abi>
//   constexpr basic_vec<T, Abi> cuda::simd::min_relu(
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
inline constexpr bool has_min_max_relu_2 = false;

template <typename Vec>
inline constexpr bool has_min_max_relu_2<
  Vec,
  cuda::std::void_t<decltype(cuda::simd::min_relu(cuda::std::declval<Vec>(), cuda::std::declval<Vec>())),
                    decltype(cuda::simd::max_relu(cuda::std::declval<Vec>(), cuda::std::declval<Vec>()))>> = true;

template <typename Vec, typename = void>
inline constexpr bool has_min_max_relu_3 = false;

template <typename Vec>
inline constexpr bool has_min_max_relu_3<
  Vec,
  cuda::std::void_t<
    decltype(cuda::simd::min_relu(cuda::std::declval<Vec>(), cuda::std::declval<Vec>(), cuda::std::declval<Vec>())),
    decltype(cuda::simd::max_relu(cuda::std::declval<Vec>(), cuda::std::declval<Vec>(), cuda::std::declval<Vec>()))>> =
  true;

template <typename T>
TEST_FUNC constexpr T scalar_max_relu(T lhs, T rhs)
{
  return cuda::std::max(cuda::std::max(lhs, rhs), T{0});
}

template <typename T>
TEST_FUNC constexpr T scalar_min_relu(T lhs, T rhs)
{
  return cuda::std::max(cuda::std::min(lhs, rhs), T{0});
}

template <typename T>
TEST_FUNC constexpr T scalar_max_relu(T a, T b, T c)
{
  return cuda::std::max(cuda::std::max(cuda::std::max(a, b), c), T{0});
}

template <typename T>
TEST_FUNC constexpr T scalar_min_relu(T a, T b, T c)
{
  return cuda::std::max(cuda::std::min(cuda::std::min(a, b), c), T{0});
}

template <typename T, int N>
TEST_FUNC constexpr void
test_values(cuda::std::array<T, N> a_values, cuda::std::array<T, N> b_values, cuda::std::array<T, N> c_values)
{
  using vec_t = fixed_size_vec<T, N>;
  vec_t a(a_values);
  vec_t b(b_values);
  vec_t c(c_values);

  static_assert(cuda::std::is_same_v<decltype(cuda::simd::max_relu(a, b)), vec_t>);
  static_assert(cuda::std::is_same_v<decltype(cuda::simd::min_relu(a, b)), vec_t>);
  static_assert(cuda::std::is_same_v<decltype(cuda::simd::max_relu(a, b, c)), vec_t>);
  static_assert(cuda::std::is_same_v<decltype(cuda::simd::min_relu(a, b, c)), vec_t>);
  static_assert(noexcept(cuda::simd::max_relu(a, b)));
  static_assert(noexcept(cuda::simd::min_relu(a, b)));
  static_assert(noexcept(cuda::simd::max_relu(a, b, c)));
  static_assert(noexcept(cuda::simd::min_relu(a, b, c)));

  vec_t max2 = cuda::simd::max_relu(a, b);
  vec_t min2 = cuda::simd::min_relu(a, b);
  vec_t max3 = cuda::simd::max_relu(a, b, c);
  vec_t min3 = cuda::simd::min_relu(a, b, c);
  for (int i = 0; i < N; ++i)
  {
    assert(max2[i] == scalar_max_relu(a_values[i], b_values[i]));
    assert(min2[i] == scalar_min_relu(a_values[i], b_values[i]));
    assert(max3[i] == scalar_max_relu(a_values[i], b_values[i], c_values[i]));
    assert(min3[i] == scalar_min_relu(a_values[i], b_values[i], c_values[i]));
  }
}

template <typename T, int N>
TEST_FUNC constexpr void test_size()
{
  constexpr auto min_val = cuda::std::numeric_limits<T>::min();
  constexpr auto max_val = cuda::std::numeric_limits<T>::max();

  cuda::std::array<T, N> a_values{min_val, max_val, T{-10}};
  cuda::std::array<T, N> b_values{max_val, min_val, T{-20}};
  cuda::std::array<T, N> c_values{T{0}, T{-1}, T{20}};
  if constexpr (N > 3)
  {
    a_values[3] = T{-5};
    b_values[3] = T{-5};
    c_values[3] = T{-5};
  }
  if constexpr (N > 4)
  {
    a_values[4] = T{0};
    b_values[4] = T{0};
    c_values[4] = T{0};
  }
  test_values<T, N>(a_values, b_values, c_values);
}

template <typename T>
TEST_FUNC constexpr void test()
{
  test_size<T, 3>();
  test_size<T, 4>();
  test_size<T, 5>();
}

TEST_FUNC constexpr bool test_all()
{
  static_assert(!has_min_max_relu_2<fixed_size_vec<unsigned, 4>>);
  static_assert(!has_min_max_relu_3<fixed_size_vec<unsigned, 4>>);
  static_assert(!has_min_max_relu_2<fixed_size_vec<float, 4>>);
  static_assert(!has_min_max_relu_3<fixed_size_vec<float, 4>>);

  test<signed char>();
  test<signed short>();
  test<signed int>();
  test<signed long>();
  test<signed long long>();
#if _CCCL_HAS_INT128()
  test<__int128_t>();
#endif // _CCCL_HAS_INT128()

  return true;
}

int main(int, char**)
{
  assert(test_all());
  static_assert(test_all());

  return 0;
}
