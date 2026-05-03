//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/__simd_>

// [simd.iterator], iterator comparisons for __simd_iterator
//
// iterator-to-iterator comparisons (==, !=, <, >, <=, >=),
// iterator-to-default_sentinel_t comparisons and noexcept guarantees.

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// iterator-to-iterator comparisons

template <typename T, int N>
TEST_FUNC constexpr void test_comparisons()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;

  Vec vec = make_iota_vec<T, N>();

  auto a = vec.begin();
  auto b = vec.begin();

  static_assert(cuda::std::is_same_v<decltype(a == b), bool>);
  static_assert(cuda::std::is_same_v<decltype(a != b), bool>);
  static_assert(cuda::std::is_same_v<decltype(a < b), bool>);
  static_assert(cuda::std::is_same_v<decltype(a > b), bool>);
  static_assert(cuda::std::is_same_v<decltype(a <= b), bool>);
  static_assert(cuda::std::is_same_v<decltype(a >= b), bool>);
  static_assert(noexcept(a == b));
  static_assert(noexcept(a != b));
  static_assert(noexcept(a < b));
  static_assert(noexcept(a > b));
  static_assert(noexcept(a <= b));
  static_assert(noexcept(a >= b));

  assert(a == b);
  assert((a != b) == false);
  assert((a < b) == false);
  assert((a > b) == false);
  assert(a <= b);
  assert(a >= b);

  if constexpr (N > 1)
  {
    auto c = a + 1;
    assert(!(a == c));
    assert(a != c);
    assert(a < c);
    assert(!(a > c));
    assert(a <= c);
    assert(!(a >= c));
    assert(c > a);
    assert(c >= a);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// sentinel comparisons and noexcept

template <typename T, int N>
TEST_FUNC constexpr void test_sentinel()
{
  using Vec      = simd::basic_vec<T, simd::fixed_size<N>>;
  using Sentinel = cuda::std::default_sentinel_t;

  Vec vec{};

  auto it  = vec.begin();
  auto end = vec.end();

  static_assert(cuda::std::is_same_v<decltype(end), Sentinel>);
  static_assert(cuda::std::is_same_v<decltype(it == end), bool>);
  static_assert(cuda::std::is_same_v<decltype(it != end), bool>);
  static_assert(cuda::std::is_same_v<decltype(end == it), bool>);
  static_assert(cuda::std::is_same_v<decltype(end != it), bool>);
  static_assert(noexcept(it == end));
  static_assert(noexcept(it != end));
  static_assert(noexcept(end == it));
  static_assert(noexcept(end != it));

  assert(!(it == end));
  assert(it != end);

  it += N;
  assert(it == end);
  assert(!(it != end));

  static_assert(cuda::std::is_same_v<decltype(it - end), typename decltype(it)::difference_type>);
  static_assert(cuda::std::is_same_v<decltype(end - it), typename decltype(it)::difference_type>);
  static_assert(noexcept(it - end));
  static_assert(noexcept(end - it));
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  test_comparisons<T, N>();
  test_sentinel<T, N>();
}

DEFINE_BASIC_VEC_TEST()
DEFINE_BASIC_VEC_TEST_RUNTIME()

int main(int, char**)
{
  assert(test());
  static_assert(test());
  assert(test_runtime());
  return 0;
}
