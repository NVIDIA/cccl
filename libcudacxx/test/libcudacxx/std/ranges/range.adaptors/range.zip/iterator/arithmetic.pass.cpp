//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// x += n;
// x + n;
// n + x;
// x -= n;
// x - n;
// x - y;
// All the arithmetic operators have the constraint `requires all-random-access<Const, Views...>;`,
// except `operator-(x, y)` which instead has the constraint
//    `requires (sized_sentinel_for<iterator_t<maybe-const<Const, Views>>,
//                                  iterator_t<maybe-const<Const, Views>>> && ...);`

#include <cuda/std/array>
#include <cuda/std/concepts>
#include <cuda/std/functional>
#include <cuda/std/ranges>

#include "../types.h"
#include "test_macros.h"

#if TEST_STD_VER >= 2020
template <class T, class U>
concept canPlusEqual = requires(T& t, U& u) { t += u; };

template <class T, class U>
concept canMinusEqual = requires(T& t, U& u) { t -= u; };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class U, class = void>
constexpr bool canPlusEqual = false;

template <class T, class U>
constexpr bool canPlusEqual<T, U, cuda::std::void_t<decltype(cuda::std::declval<T&>() += cuda::std::declval<U&>())>> =
  true;

template <class T, class U, class = void>
constexpr bool canMinusEqual = false;

template <class T, class U>
constexpr bool canMinusEqual<T, U, cuda::std::void_t<decltype(cuda::std::declval<T&>() -= cuda::std::declval<U&>())>> =
  true;
#endif // TEST_STD_VER <= 2017

__host__ __device__ constexpr bool test()
{
  int buffer1[5] = {1, 2, 3, 4, 5};
  int buffer2[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  SizedRandomAccessView a{buffer1};
  static_assert(cuda::std::ranges::random_access_range<decltype(a)>);
  cuda::std::array<double, 5> b{4.1, 3.2, 4.3, 0.1, 0.2};
  static_assert(cuda::std::ranges::contiguous_range<decltype(b)>);
  {
    // operator+(x, n) and operator+=
    cuda::std::ranges::zip_view v(a, cuda::std::span{b});
    auto it1 = v.begin();

    auto it2      = it1 + 3;
    auto [x2, y2] = *it2;
    assert(&x2 == &(a[3]));
    assert(&y2 == &(b[3]));

    auto it3      = 3 + it1;
    auto [x3, y3] = *it3;
    assert(&x3 == &(a[3]));
    assert(&y3 == &(b[3]));

    it1 += 3;
    assert(it1 == it2);
    auto [x1, y1] = *it2;
    assert(&x1 == &(a[3]));
    assert(&y1 == &(b[3]));

    using Iter = decltype(it1);
    static_assert(canPlusEqual<Iter, intptr_t>);
  }

  {
    // operator-(x, n) and operator-=
    cuda::std::ranges::zip_view v(a, cuda::std::span{b});
    auto it1 = v.end();

    auto it2      = it1 - 3;
    auto [x2, y2] = *it2;
    assert(&x2 == &(a[2]));
    assert(&y2 == &(b[2]));

    it1 -= 3;
    assert(it1 == it2);
    auto [x1, y1] = *it2;
    assert(&x1 == &(a[2]));
    assert(&y1 == &(b[2]));

    using Iter = decltype(it1);
    static_assert(canMinusEqual<Iter, intptr_t>);
  }

  {
    // operator-(x, y)
    cuda::std::ranges::zip_view v(a, cuda::std::span{b});
    assert((v.end() - v.begin()) == 5);

    auto it1 = v.begin() + 2;
    auto it2 = v.end() - 1;
    assert((it1 - it2) == -2);
  }

  {
    // in this case sentinel is computed by getting each of the underlying sentinels, so the distance
    // between begin and end for each of the underlying iterators can be different
    cuda::std::ranges::zip_view v{ForwardSizedView(buffer1), ForwardSizedView(buffer2)};
    using View = decltype(v);
    static_assert(cuda::std::ranges::common_range<View>);
    static_assert(!cuda::std::ranges::random_access_range<View>);

    auto it1 = v.begin();
    auto it2 = v.end();
    // it1 : <buffer1 + 0, buffer2 + 0>
    // it2 : <buffer1 + 5, buffer2 + 9>
    assert((it1 - it2) == -5);
    assert((it2 - it1) == 5);
  }

  {
    // One of the ranges is not random access
    cuda::std::ranges::zip_view v(a, cuda::std::span{b}, ForwardSizedView{buffer1});
    using Iter = decltype(v.begin());
    static_assert(!cuda::std::invocable<cuda::std::plus<>, Iter, intptr_t>);
    static_assert(!cuda::std::invocable<cuda::std::plus<>, intptr_t, Iter>);
    static_assert(!canPlusEqual<Iter, intptr_t>);
    static_assert(!cuda::std::invocable<cuda::std::minus<>, Iter, intptr_t>);
    static_assert(cuda::std::invocable<cuda::std::minus<>, Iter, Iter>);
    static_assert(!canMinusEqual<Iter, intptr_t>);
  }

  {
    // One of the ranges does not have sized sentinel
    cuda::std::ranges::zip_view v(a, cuda::std::span{b}, InputCommonView{buffer1});
    using Iter = decltype(v.begin());
    static_assert(!cuda::std::invocable<cuda::std::minus<>, Iter, Iter>);
  }

  return true;
}

int main(int, char**)
{
  test();
#if defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // defined(_LIBCUDACXX_ADDRESSOF)

  return 0;
}
