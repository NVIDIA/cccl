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

// friend constexpr void iter_swap(const iterator& x, const iterator& y);

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "../types.h"

using NonSwappableView = BufferView<copying_iterator<int*>>;
static_assert(cuda::std::ranges::input_range<NonSwappableView>);
static_assert(!cuda::std::indirectly_swappable<cuda::std::ranges::iterator_t<NonSwappableView>>);

__host__ __device__ constexpr bool test()
{
  int buffer[4][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};

  {
    cuda::std::ranges::join_view jv(buffer);
    auto iter1 = jv.begin();
    auto iter2 = cuda::std::next(jv.begin());
    assert(buffer[0][0] == 1);
    assert(buffer[0][1] == 2);
    cuda::std::ranges::iter_swap(iter1, iter2);
    assert(buffer[0][0] == 2);
    assert(buffer[0][1] == 1);

    static_assert(noexcept(cuda::std::ranges::iter_swap(iter1, iter2)));
  }

  {
    // iter_move calls inner's iter_swap
    IterMoveSwapAwareView inners[1] = {buffer[0]};
    cuda::std::ranges::join_view jv(inners);
    auto it1 = jv.begin();
    auto it2 = cuda::std::ranges::next(it1);

    const auto& iter_swap_called_times = jv.base().begin()->iter_swap_called;

    assert(iter_swap_called_times == 0);
    assert(buffer[0][0] == 2);
    assert(buffer[0][1] == 1);

    cuda::std::ranges::iter_swap(it1, it2);

    assert(buffer[0][0] == 1);
    assert(buffer[0][1] == 2);
    assert(iter_swap_called_times == 1);
  }

  {
    // LWG3517 `join_view::iterator`'s `iter_swap` is underconstrained
    // `iter_swap` should not be defined if Inner's iterator does not indirectly_swappable
    NonSwappableView inners[2] = {buffer[0], buffer[1]};
    cuda::std::ranges::join_view jv(inners);
    using Iter = cuda::std::ranges::iterator_t<decltype(jv)>;
    static_assert(!cuda::std::indirectly_swappable<Iter>);
  }
  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)

  return 0;
}
