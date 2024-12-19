//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// template<class T>
// class empty_view;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_macros.h"

template <class T>
__host__ __device__ constexpr void testType()
{
  static_assert(cuda::std::ranges::range<cuda::std::ranges::empty_view<T>>);
  static_assert(cuda::std::ranges::range<const cuda::std::ranges::empty_view<T>>);
  static_assert(cuda::std::ranges::view<cuda::std::ranges::empty_view<T>>);

  cuda::std::ranges::empty_view<T> empty{};

  assert(empty.begin() == nullptr);
  assert(empty.end() == nullptr);
  assert(empty.data() == nullptr);
  assert(empty.size() == 0);
  assert(empty.empty() == true);

  assert(cuda::std::ranges::begin(empty) == nullptr);
  assert(cuda::std::ranges::end(empty) == nullptr);
  assert(cuda::std::ranges::data(empty) == nullptr);
  assert(cuda::std::ranges::size(empty) == 0);
  assert(cuda::std::ranges::empty(empty) == true);
}

struct Empty
{};
struct BigType
{
  char buff[8];
};

#if TEST_STD_VER >= 2020
template <class T>
concept ValidEmptyView = requires { typename cuda::std::ranges::empty_view<T>; };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
constexpr bool ValidEmptyView = false;

template <class T>
constexpr bool ValidEmptyView<T, cuda::std::void_t<cuda::std::ranges::empty_view<T>>> = true;
#endif // TEST_STD_VER <= 2017

__host__ __device__ constexpr bool test()
{
  // Not objects:
  static_assert(!ValidEmptyView<int&>);
  static_assert(!ValidEmptyView<void>);

  testType<int>();
  testType<const int>();
  testType<int*>();
  testType<Empty>();
  testType<const Empty>();
  testType<BigType>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
