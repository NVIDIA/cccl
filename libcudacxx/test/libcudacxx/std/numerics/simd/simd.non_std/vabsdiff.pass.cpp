//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/__simd/abs_diff.h>

// template<class T, class Abi>
//   constexpr basic_vec<make_unsigned_t<T>, Abi> cuda::simd::abs_diff(
//     const basic_vec<T, Abi>& lhs, const basic_vec<T, Abi>& rhs) noexcept;

#include <cuda/simd>
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
inline constexpr bool has_abs_diff = false;

template <typename Vec>
inline constexpr bool has_abs_diff<
  Vec,
  cuda::std::void_t<decltype(cuda::simd::abs_diff(cuda::std::declval<Vec>(), cuda::std::declval<Vec>()))>> = true;

template <typename T>
TEST_FUNC constexpr cuda::std::make_unsigned_t<T> scalar_abs_diff(T lhs, T rhs)
{
  using ResultT = cuda::std::make_unsigned_t<T>;
  ResultT lhs_u = static_cast<ResultT>(lhs);
  ResultT rhs_u = static_cast<ResultT>(rhs);
  return (lhs < rhs) ? (rhs_u - lhs_u) : (lhs_u - rhs_u);
}

template <typename T, int N>
TEST_FUNC constexpr void test_values(cuda::std::array<T, N> lhs_values, cuda::std::array<T, N> rhs_values)
{
  using Vec       = simd::basic_vec<T, simd::fixed_size<N>>;
  using ResultVec = simd::basic_vec<cuda::std::make_unsigned_t<T>, simd::fixed_size<N>>;
  Vec lhs(lhs_values);
  Vec rhs(rhs_values);

  static_assert(cuda::std::is_same_v<decltype(cuda::simd::abs_diff(lhs, rhs)), ResultVec>);
  static_assert(noexcept(cuda::simd::abs_diff(lhs, rhs)));

  ResultVec result = cuda::simd::abs_diff(lhs, rhs);
  for (int i = 0; i < N; ++i)
  {
    assert(result[i] == scalar_abs_diff(lhs_values[i], rhs_values[i]));
  }
}

template <typename T, int N>
TEST_FUNC constexpr void test_size()
{
  constexpr auto min_val = cuda::std::numeric_limits<T>::min();
  constexpr auto max_val = cuda::std::numeric_limits<T>::max();

  if constexpr (cuda::std::is_signed_v<T>)
  {
    cuda::std::array<T, N> lhs_values{min_val, max_val, T{-10}};
    cuda::std::array<T, N> rhs_values{max_val, min_val, T{20}};
    if constexpr (N > 3)
    {
      lhs_values[3] = T{-5};
      rhs_values[3] = T{-5};
    }
    test_values<T, N>(lhs_values, rhs_values);
  }
  else
  {
    cuda::std::array<T, N> lhs_values{min_val, max_val, T{10}};
    cuda::std::array<T, N> rhs_values{max_val, min_val, T{20}};
    if constexpr (N > 3)
    {
      lhs_values[3] = T{5};
      rhs_values[3] = T{5};
    }
    test_values<T, N>(lhs_values, rhs_values);
  }
}

template <typename T>
TEST_FUNC constexpr void test()
{
  test_size<T, 4>();
  test_size<T, 3>();
}

TEST_FUNC constexpr bool test_all()
{
  static_assert(!has_abs_diff<fixed_size_vec<float, 4>>);

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
