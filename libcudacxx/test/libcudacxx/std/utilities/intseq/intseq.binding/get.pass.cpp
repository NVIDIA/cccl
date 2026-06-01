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
//   constexpr T get(integer_sequence<T, Values...>) noexcept;

#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/utility>

#include "test_macros.h"

template <class Seq, cuda::std::size_t I>
TEST_FUNC constexpr void test(typename Seq::value_type ref_val)
{
  // Test Seq.
  {
    Seq seq{};
    static_assert(cuda::std::is_same_v<cuda::std::tuple_element_t<I, Seq>, decltype(cuda::std::get<I>(seq))>);
    static_assert(noexcept(cuda::std::get<I>(seq)));

    const auto v1 = cuda::std::get<I>(seq);
    assert(v1 == ref_val);

    // Test ADL.
#if TEST_STD_VER >= 2020
    const auto v2 = get<I>(seq);
    assert(v2 == ref_val);
#endif // TEST_STD_VER >= 2020
  }

  // Test const Seq.
  {
    const Seq seq{};
    static_assert(cuda::std::is_same_v<cuda::std::tuple_element_t<I, Seq>, decltype(cuda::std::get<I>(seq))>);
    static_assert(noexcept(cuda::std::get<I>(seq)));

    const auto v1 = cuda::std::get<I>(seq);
    assert(v1 == ref_val);

    // Test ADL.
#if TEST_STD_VER >= 2020
    const auto v2 = get<I>(seq);
    assert(v2 == ref_val);
#endif // TEST_STD_VER >= 2020
  }
}

template <class T, T... Vs>
TEST_FUNC constexpr void test()
{
  cuda::static_for<sizeof...(Vs)>([](auto i) {
    constexpr T values[(sizeof...(Vs) > 0) ? sizeof...(Vs) : 1]{Vs...};
    test<cuda::std::integer_sequence<T, Vs...>, i()>(values[i()]);
  });
}

TEST_FUNC constexpr bool test()
{
  test<cuda::std::integer_sequence<signed char, -1>>();
  test<cuda::std::integer_sequence<unsigned, 1, 2, 3>>();
  test<cuda::std::integer_sequence<unsigned long long, 1, 2, 3, 8098098>>();

  const auto [v1, v2, v3] = cuda::std::integer_sequence<unsigned, 1, 2, 3>{};
  assert(v1 == 1);
  assert(v2 == 2);
  assert(v3 == 3);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
