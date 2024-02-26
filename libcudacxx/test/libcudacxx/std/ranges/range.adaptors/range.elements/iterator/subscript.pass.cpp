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

// constexpr decltype(auto) operator[](difference_type n) const
//   requires random_access_range<Base>

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/tuple>

#include "test_iterators.h"
#include "test_macros.h"

#if TEST_STD_VER >= 2020
template <class T, class U>
concept CanSubscript = requires(T t, U u) { t[u]; };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class U, class = void>
inline constexpr bool CanSubscript = false;

template <class T, class U>
inline constexpr bool CanSubscript<T, U, cuda::std::void_t<decltype(cuda::std::declval<T>()[cuda::std::declval<U>()])>> =
  true;
#endif // TEST_STD_VER <= 2017
template <class BaseRange>
using ElemIter = cuda::std::ranges::iterator_t<cuda::std::ranges::elements_view<BaseRange, 0>>;

using RandomAccessRange = cuda::std::ranges::subrange<cuda::std::tuple<int>*>;
static_assert(cuda::std::ranges::random_access_range<RandomAccessRange>);

static_assert(CanSubscript<ElemIter<RandomAccessRange>, int>);

using BidiRange = cuda::std::ranges::subrange<bidirectional_iterator<cuda::std::tuple<int>*>>;
static_assert(!cuda::std::ranges::random_access_range<BidiRange>);

static_assert(!CanSubscript<ElemIter<BidiRange>, int>);

__host__ __device__ constexpr bool test()
{
  {
    // reference
    cuda::std::tuple<int> ts[] = {{1}, {2}, {3}, {4}};
    auto ev                    = ts | cuda::std::views::elements<0>;
    auto it                    = ev.begin();

    assert(&it[0] == &*it);
    assert(&it[2] == &*(it + 2));

    static_assert(cuda::std::is_same_v<decltype(it[2]), int&>);
  }

  {
    // value
    auto ev = cuda::std::views::iota(0, 5) | cuda::std::views::transform([](int i) {
                return cuda::std::tuple<int>{i};
              })
            | cuda::std::views::elements<0>;
    auto it = ev.begin();
    assert(it[0] == *it);
    assert(it[2] == *(it + 2));
    assert(it[4] == *(it + 4));

    static_assert(cuda::std::is_same_v<decltype(it[2]), int>);
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test());
#endif // TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)

  return 0;
}
