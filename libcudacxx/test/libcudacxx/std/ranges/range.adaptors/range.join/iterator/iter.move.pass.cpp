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

// friend constexpr decltype(auto) iter_move(const iterator& i);

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "../types.h"

__host__ __device__ constexpr bool test()
{
  int buffer[4][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};

  {
    cuda::std::ranges::join_view jv(buffer);
    assert(cuda::std::ranges::iter_move(jv.begin()) == 1);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::iter_move(jv.begin())), int&&>);

    static_assert(noexcept(cuda::std::ranges::iter_move(cuda::std::declval<decltype(jv.begin())>())));
  }

  {
    // iter_move calls inner's iter_move and calls
    // iter_move on the correct inner iterator
    IterMoveSwapAwareView inners[2] = {buffer[0], buffer[1]};
    cuda::std::ranges::join_view jv(inners);
    auto it = jv.begin();

    const auto& iter_move_called_times1 = jv.base().begin()->iter_move_called;
    const auto& iter_move_called_times2 = cuda::std::next(jv.base().begin())->iter_move_called;
    assert(iter_move_called_times1 == 0);
    assert(iter_move_called_times2 == 0);

    decltype(auto) x = cuda::std::ranges::iter_move(it);
    static_assert(cuda::std::same_as<decltype(x), cuda::std::pair<int&&, int&&>>);
    assert(cuda::std::get<0>(x) == 1);
    assert(iter_move_called_times1 == 1);
    assert(iter_move_called_times2 == 0);

    auto it2 = cuda::std::ranges::next(it, 4);

    decltype(auto) y = cuda::std::ranges::iter_move(it2);
    static_assert(cuda::std::same_as<decltype(y), cuda::std::pair<int&&, int&&>>);
    assert(cuda::std::get<0>(y) == 5);
    assert(iter_move_called_times1 == 1);
    assert(iter_move_called_times2 == 1);
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
