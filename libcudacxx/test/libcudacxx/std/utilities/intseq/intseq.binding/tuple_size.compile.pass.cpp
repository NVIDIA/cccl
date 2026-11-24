//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/utility>

// template<class T, T... Values>
//   struct tuple_size<integer_sequence<T, Values...>>;

#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

template <class T, T... Vs>
TEST_FUNC constexpr bool test_tuple_size()
{
  using Seq               = cuda::std::integer_sequence<T, Vs...>;
  constexpr auto ref_size = sizeof...(Vs);

  // Test std::tuple_size.
  static_assert(std::tuple_size<Seq>::value == ref_size);
  static_assert(std::tuple_size<const Seq>::value == ref_size);

  // Test cuda::std::tuple_size.
  static_assert(cuda::std::tuple_size<Seq>::value == ref_size);
  static_assert(cuda::std::tuple_size<const Seq>::value == ref_size);

  // Test cuda::std::tuple_size_v.
  static_assert(cuda::std::tuple_size_v<Seq> == ref_size);
  static_assert(cuda::std::tuple_size_v<const Seq> == ref_size);

  return true;
}

static_assert(test_tuple_size<int>());
static_assert(test_tuple_size<signed char, -1>());
static_assert(test_tuple_size<unsigned, 1, 2, 3>());
static_assert(test_tuple_size<unsigned long long, 1, 2, 3, 8098098>());

int main(int, char**)
{
  return 0;
}
