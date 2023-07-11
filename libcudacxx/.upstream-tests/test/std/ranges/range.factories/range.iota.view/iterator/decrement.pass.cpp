//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// constexpr iterator& operator--() requires decrementable<W>;
// constexpr iterator operator--(int) requires decrementable<W>;

#include <cuda/std/ranges>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "../types.h"

#if TEST_STD_VER > 17
template<class T>
concept Decrementable =
  requires(T i) {
    --i;
  } ||
  requires(T i) {
    i--;
  };
#else
template<class T, class = void>
inline constexpr bool Decrementable1 = false;

template<class T>
inline constexpr bool Decrementable1<T, cuda::std::void_t<decltype(--cuda::std::declval<T>())>> = true;

template<class T, class = void>
inline constexpr bool Decrementable2 = false;
template<class T>
inline constexpr bool Decrementable2<T, cuda::std::void_t<decltype(cuda::std::declval<T>()--)>> = true;

template<class T>
inline constexpr bool Decrementable = Decrementable1<T> || Decrementable2<T>;
#endif

__host__ __device__ constexpr bool test() {
  {
    cuda::std::ranges::iota_view<int> io(0);
    auto iter1 = cuda::std::next(io.begin());
    auto iter2 = cuda::std::next(io.begin());
    assert(iter1 == iter2);
    assert(--iter1 != iter2--);
    assert(iter1 == iter2);

    static_assert(!cuda::std::is_reference_v<decltype(iter2--)>);
    static_assert( cuda::std::is_reference_v<decltype(--iter2)>);
    static_assert(cuda::std::same_as<cuda::std::remove_reference_t<decltype(--iter2)>, decltype(iter2--)>);
  }
  {
    cuda::std::ranges::iota_view io(SomeInt(0));
    auto iter1 = cuda::std::next(io.begin());
    auto iter2 = cuda::std::next(io.begin());
    assert(iter1 == iter2);
    assert(--iter1 != iter2--);
    assert(iter1 == iter2);

    static_assert(!cuda::std::is_reference_v<decltype(iter2--)>);
    static_assert( cuda::std::is_reference_v<decltype(--iter2)>);
    static_assert(cuda::std::same_as<cuda::std::remove_reference_t<decltype(--iter2)>, decltype(iter2--)>);
  }

  static_assert(!Decrementable<cuda::std::ranges::iota_view<NotDecrementable>>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");

  return 0;
}
