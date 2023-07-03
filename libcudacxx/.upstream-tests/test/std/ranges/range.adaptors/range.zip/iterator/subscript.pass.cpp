//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// constexpr auto operator[](difference_type n) const requires
//        all_random_access<Const, Views...>

#include <cuda/std/ranges>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "../types.h"

#if TEST_STD_VER > 17
template <class Iter>
concept canSubscript = requires(Iter it) { it[0]; };
#else
template<class Iter, class = void>
constexpr bool canSubscript = false;

template<class Iter>
constexpr bool canSubscript<Iter, cuda::std::void_t<decltype(cuda::std::declval<Iter>()[0])>> = true;
#endif

__host__ __device__ constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    // random_access_range
    cuda::std::ranges::zip_view v(SizedRandomAccessView{buffer}, cuda::std::views::iota(0));
    auto it = v.begin();
    assert(it[0] == *it);
    assert(it[2] == *(it + 2));
    assert(it[4] == *(it + 4));

    static_assert(cuda::std::is_same_v<decltype(it[2]), cuda::std::pair<int&, int>>);
  }

  {
    // contiguous_range
    cuda::std::ranges::zip_view v(ContiguousCommonView{buffer}, ContiguousCommonView{buffer});
    auto it = v.begin();
    assert(it[0] == *it);
    assert(it[2] == *(it + 2));
    assert(it[4] == *(it + 4));

    static_assert(cuda::std::is_same_v<decltype(it[2]), cuda::std::pair<int&, int&>>);
  }

  {
    // non random_access_range
    cuda::std::ranges::zip_view v(BidiCommonView{buffer});
    auto iter = v.begin();
    static_assert(!canSubscript<decltype(iter)>);
    unused(iter);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");

  return 0;
}
