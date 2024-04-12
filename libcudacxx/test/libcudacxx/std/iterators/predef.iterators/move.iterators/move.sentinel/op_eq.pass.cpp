//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <cuda/std/iterator>

// move_sentinel

// template <class Iter, class Sent>
//   constexpr bool operator==(const move_iterator<Iter>& x, const move_sentinel<Sent>& y);

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

#if TEST_STD_VER > 2017
template <class T, class U>
concept HasEquals = requires(T t, U u) { t == u; };
template <class T, class U>
concept HasNotEquals = requires(T t, U u) { t != u; };
template <class T, class U>
concept HasLess = requires(T t, U u) { t < u; };
#else
template <class T, class U, class = void>
constexpr bool HasEquals = false;

template <class T, class U>
constexpr bool HasEquals<T, U, cuda::std::void_t<decltype(cuda::std::declval<T>() == cuda::std::declval<U>())>> = true;

template <class T, class U, class = void>
constexpr bool HasNotEquals = false;

template <class T, class U>
constexpr bool HasNotEquals<T, U, cuda::std::void_t<decltype(cuda::std::declval<T>() != cuda::std::declval<U>())>> =
  true;

template <class T, class U, class = void>
constexpr bool HasLess = false;

template <class T, class U>
constexpr bool HasLess<T, U, cuda::std::void_t<decltype(cuda::std::declval<T>() < cuda::std::declval<U>())>> = true;
#endif

static_assert(!HasEquals<cuda::std::move_iterator<int*>, cuda::std::move_sentinel<char*>>);
static_assert(!HasNotEquals<cuda::std::move_iterator<int*>, cuda::std::move_sentinel<char*>>);
static_assert(!HasLess<cuda::std::move_iterator<int*>, cuda::std::move_sentinel<char*>>);

static_assert(HasEquals<cuda::std::move_iterator<int*>, cuda::std::move_sentinel<const int*>>);
static_assert(HasNotEquals<cuda::std::move_iterator<int*>, cuda::std::move_sentinel<const int*>>);
static_assert(!HasLess<cuda::std::move_iterator<int*>, cuda::std::move_sentinel<const int*>>);

static_assert(HasEquals<cuda::std::move_iterator<const int*>, cuda::std::move_sentinel<int*>>);
static_assert(HasNotEquals<cuda::std::move_iterator<const int*>, cuda::std::move_sentinel<int*>>);
static_assert(!HasLess<cuda::std::move_iterator<const int*>, cuda::std::move_sentinel<int*>>);

template <class It>
__host__ __device__ constexpr void test_one()
{
  char s[]         = "abc";
  const auto it    = cuda::std::move_iterator<It>(It(s));
  const auto sent1 = cuda::std::move_sentinel<sentinel_wrapper<It>>(sentinel_wrapper<It>(It(s)));
  const auto sent2 = cuda::std::move_sentinel<sentinel_wrapper<It>>(sentinel_wrapper<It>(It(s + 1)));
  ASSERT_SAME_TYPE(decltype(it == sent1), bool);
  assert((it == sent1));
  assert(!(it != sent1));
  assert(!(it == sent2));
  assert((it != sent2));
  assert((sent1 == it));
  assert(!(sent1 != it));
  assert(!(sent2 == it));
  assert((sent2 != it));
  static_assert(!HasEquals<decltype(sent1), decltype(sent1)>);
  static_assert(!HasLess<decltype(sent1), decltype(sent1)>);
}

__host__ __device__ constexpr bool test()
{
  test_one<cpp17_input_iterator<char*>>();
  test_one<cpp20_input_iterator<char*>>();
  test_one<forward_iterator<char*>>();
  test_one<bidirectional_iterator<char*>>();
  test_one<random_access_iterator<char*>>();
  test_one<contiguous_iterator<char*>>();
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  test_one<three_way_contiguous_iterator<char*>>();
#endif
  test_one<char*>();
  test_one<const char*>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
