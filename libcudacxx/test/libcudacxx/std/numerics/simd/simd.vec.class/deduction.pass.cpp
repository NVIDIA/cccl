//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// CTAD is unsupported on MSVC.

// UNSUPPORTED: msvc

// <cuda/std/__simd_>

// [simd.ctor] deduction guides
//
// basic_vec(Range&&, Ts...) -> basic_vec<range_value_t<Range>, deduce-abi-t<...>>;
// basic_vec(basic_mask<Bytes, Abi>) -> basic_vec<integer-from<Bytes>, Abi>;

#include <cuda/std/__simd_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/span>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// deduction from range

template <typename T, int N>
TEST_FUNC constexpr void test_range_deduction()
{
  cuda::std::array<T, N> arr{};
  for (int i = 0; i < N; ++i)
  {
    arr[i] = static_cast<T>(i);
  }
  simd::basic_vec vec(arr);
  static_assert(cuda::std::is_same_v<typename decltype(vec)::value_type, T>);
  static_assert(decltype(vec)::size() == N);
  for (int i = 0; i < N; ++i)
  {
    assert(vec[i] == static_cast<T>(i));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// deduction from fixed-extent span

template <typename T, int N>
TEST_FUNC constexpr void test_span_deduction()
{
  cuda::std::array<T, N> arr{};
  for (int i = 0; i < N; ++i)
  {
    arr[i] = static_cast<T>(i);
  }

  const cuda::std::span<T, N> values(arr);
  simd::basic_vec vec(values);
  static_assert(cuda::std::is_same_v<typename decltype(vec)::value_type, T>);
  static_assert(decltype(vec)::size() == N);
  for (int i = 0; i < N; ++i)
  {
    assert(vec[i] == static_cast<T>(i));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// deduction from basic_mask

template <int Bytes, int N>
TEST_FUNC constexpr void test_mask_deduction()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  Mask mask(true);
  simd::basic_vec vec(mask);
  static_assert(decltype(vec)::size() == N);
  for (int i = 0; i < N; ++i)
  {
    assert(vec[i] == 1);
  }
}

//----------------------------------------------------------------------------------------------------------------------

TEST_FUNC constexpr bool test_deduction()
{
  test_range_deduction<int, 1>();
  test_range_deduction<int, 4>();
  test_range_deduction<short, 4>();
  test_range_deduction<long long, 4>();
  test_span_deduction<int, 1>();
  test_span_deduction<int, 4>();
  test_span_deduction<short, 4>();
  test_span_deduction<long long, 4>();

  test_mask_deduction<1, 1>();
  test_mask_deduction<1, 4>();
  test_mask_deduction<4, 4>();
  return true;
}

int main(int, char**)
{
  assert(test_deduction());
  static_assert(test_deduction());
  return 0;
}
