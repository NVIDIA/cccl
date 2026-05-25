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

// [simd.iterator], iterator arithmetic for __simd_iterator
//
// pre/post increment and decrement, compound assignment (+=, -=),
// binary arithmetic (it + n, n + it, it - n, it - it), and differences
// between iterators and default_sentinel_t.

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/memory>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// pre-increment

template <typename T, int N>
TEST_FUNC constexpr void test_pre_increment()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Iter = typename Vec::iterator;
  Vec vec    = make_iota_vec<T, N>();

  auto it    = vec.begin();
  auto&& ref = ++it;
  static_assert(cuda::std::is_same_v<decltype(ref), Iter&>);
  static_assert(noexcept(++it));
  assert(cuda::std::addressof(ref) == cuda::std::addressof(it));
  if constexpr (N > 1)
  {
    assert(*it == static_cast<T>(1));
  }
  else
  {
    assert(it == cuda::std::default_sentinel);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// post-increment

template <typename T, int N>
TEST_FUNC constexpr void test_post_increment()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Iter = typename Vec::iterator;
  Vec vec    = make_iota_vec<T, N>();

  auto it2   = vec.begin();
  auto begin = it2;
  auto old   = it2++;
  static_assert(cuda::std::is_same_v<decltype(it2++), Iter>);
  static_assert(noexcept(it2++));
  assert(old == begin);
  if constexpr (N > 1)
  {
    assert(*it2 == static_cast<T>(1));
  }
  else
  {
    assert(it2 == cuda::std::default_sentinel);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// pre-decrement

template <typename T, int N>
TEST_FUNC constexpr void test_pre_decrement()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Iter = typename Vec::iterator;
  Vec vec    = make_iota_vec<T, N>();

  auto it3    = vec.begin() + 1;
  auto&& ref3 = --it3;
  static_assert(cuda::std::is_same_v<decltype(ref3), Iter&>);
  static_assert(noexcept(--it3));
  assert(cuda::std::addressof(ref3) == cuda::std::addressof(it3));
  assert(*it3 == static_cast<T>(0));
}

//----------------------------------------------------------------------------------------------------------------------
// post-decrement

template <typename T, int N>
TEST_FUNC constexpr void test_post_decrement()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Iter = typename Vec::iterator;
  Vec vec    = make_iota_vec<T, N>();

  auto it4  = vec.begin() + 1;
  auto old2 = it4--;
  static_assert(cuda::std::is_same_v<decltype(it4--), Iter>);
  static_assert(noexcept(it4--));
  assert(old2 == vec.begin() + 1);
  assert(*it4 == static_cast<T>(0));
}

//----------------------------------------------------------------------------------------------------------------------
// compound assignment

template <typename T, int N>
TEST_FUNC constexpr void test_compound_assignment()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Iter = typename Vec::iterator;
  Vec vec    = make_iota_vec<T, N>();

  auto it     = vec.begin();
  auto&& ref1 = (it += N);
  static_assert(cuda::std::is_same_v<decltype(ref1), Iter&>);
  static_assert(noexcept(it += N));
  assert(cuda::std::addressof(ref1) == cuda::std::addressof(it));
  assert(it == cuda::std::default_sentinel);

  auto&& ref2 = (it -= N);
  static_assert(cuda::std::is_same_v<decltype(ref2), Iter&>);
  static_assert(noexcept(it -= N));
  assert(cuda::std::addressof(ref2) == cuda::std::addressof(it));
  assert(*it == static_cast<T>(0));
}

//----------------------------------------------------------------------------------------------------------------------
// binary arithmetic

template <typename T, int N>
TEST_FUNC constexpr void test_arithmetic()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Iter = typename Vec::iterator;
  using Diff = typename Iter::difference_type;
  Vec vec    = make_iota_vec<T, N>();

  auto it = vec.begin();

  auto it2 = it + N;
  static_assert(cuda::std::is_same_v<decltype(it + N), Iter>);
  static_assert(noexcept(it + N));
  assert(it2 == cuda::std::default_sentinel);
  // operator+(N)
  {
    auto it3 = N + it;
    static_assert(cuda::std::is_same_v<decltype(N + it), Iter>);
    static_assert(noexcept(N + it));
    assert(it3 == cuda::std::default_sentinel);
  }
  // operator-(N)
  {
    auto it4 = it2 - N;
    static_assert(cuda::std::is_same_v<decltype(it2 - N), Iter>);
    static_assert(noexcept(it2 - N));
    assert(*it4 == static_cast<T>(0));
  }
  // operator-
  {
    static_assert(cuda::std::is_same_v<decltype(it2 - it), Diff>);
    static_assert(noexcept(it2 - it));
    assert(it2 - it == N);
    assert(it - it2 == -N);
  }
  // operator-(default_sentinel_t)
  {
    static_assert(cuda::std::is_same_v<decltype(it - cuda::std::default_sentinel), Diff>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::default_sentinel - it), Diff>);
    static_assert(noexcept(it - cuda::std::default_sentinel));
    static_assert(noexcept(cuda::std::default_sentinel - it));
    assert(it - cuda::std::default_sentinel == -N);
    assert(cuda::std::default_sentinel - it == N);
    assert(it2 - cuda::std::default_sentinel == 0);
    assert(cuda::std::default_sentinel - it2 == 0);
  }
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  test_pre_increment<T, N>();
  test_post_increment<T, N>();
  test_pre_decrement<T, N>();
  test_post_decrement<T, N>();
  test_compound_assignment<T, N>();
  test_arithmetic<T, N>();
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
