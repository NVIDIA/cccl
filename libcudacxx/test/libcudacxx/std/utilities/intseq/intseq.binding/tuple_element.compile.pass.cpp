//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/utility>

// template<size_t I, class T, T... Values>
//   struct tuple_element<I, integer_sequence<T, Values...>>;
// template<size_t I, class T, T... Values>
//   struct tuple_element<I, const integer_sequence<T, Values...>>;

#include <cuda/std/type_traits>
#include <cuda/utility>

#include "test_macros.h"

template <class T, T... Vs>
struct TestTupleElement
{
  template <cuda::std::size_t I>
  TEST_FUNC constexpr void operator()(cuda::std::integral_constant<cuda::std::size_t, I>) const
  {
    using Seq = cuda::std::integer_sequence<T, Vs...>;

    // Test std::tuple_element.
    static_assert(cuda::std::is_same_v<T, typename std::tuple_element<I, Seq>::type>);
    static_assert(cuda::std::is_same_v<T, typename std::tuple_element<I, const Seq>::type>);

    // Test cuda::std::tuple_element.
    static_assert(cuda::std::is_same_v<T, typename cuda::std::tuple_element<I, Seq>::type>);
    static_assert(cuda::std::is_same_v<T, typename cuda::std::tuple_element<I, const Seq>::type>);

    // Test cuda::std::tuple_element_t.
    static_assert(cuda::std::is_same_v<T, cuda::std::tuple_element_t<I, Seq>>);
    static_assert(cuda::std::is_same_v<T, cuda::std::tuple_element_t<I, const Seq>>);
  }
};

template <class T, T... Vs>
TEST_FUNC constexpr bool test_tuple_element()
{
  cuda::static_for<sizeof...(Vs)>(TestTupleElement<T, Vs...>{});

  return true;
}

static_assert(test_tuple_element<signed char, -1>());
static_assert(test_tuple_element<unsigned, 1, 2, 3>());
static_assert(test_tuple_element<unsigned long long, 1, 2, 3, 8098098>());

int main(int, char**)
{
  return 0;
}
