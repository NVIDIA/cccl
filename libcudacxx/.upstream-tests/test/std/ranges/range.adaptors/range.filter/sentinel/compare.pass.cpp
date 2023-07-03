//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// friend constexpr bool operator==(iterator const&, sentinel const&);

#include <cuda/std/ranges>

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "test_macros.h"
#include "../types.h"

template <class Iterator, class Sentinel = sentinel_wrapper<Iterator>>
__host__ __device__ constexpr void test() {
  using View = minimal_view<Iterator, Sentinel>;

  cuda::std::array<int, 5> array{0, 1, 2, 3, 4};

  {
    View v(Iterator(array.begin()), Sentinel(Iterator(array.end())));
    cuda::std::ranges::filter_view<View, AlwaysTrue> view(cuda::std::move(v), AlwaysTrue{});
    auto const it = view.begin();
    auto const sent = view.end();
    decltype(auto) result = (it == sent);
    static_assert(cuda::std::same_as<decltype(result), bool>);
    assert(!result);
  }
  {
    View v(Iterator(array.begin()), Sentinel(Iterator(array.end())));
    cuda::std::ranges::filter_view<View, AlwaysFalse> view(cuda::std::move(v), AlwaysFalse{});
    auto const it = view.begin();
    auto const sent = view.end();
    decltype(auto) result = (it == sent);
    static_assert(cuda::std::same_as<decltype(result), bool>);
    assert(result);
  }
}

__host__ __device__ constexpr bool tests() {
  test<cpp17_input_iterator<int*>>();
  test<cpp20_input_iterator<int*>>();
  test<forward_iterator<int*>>();
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<contiguous_iterator<int*>>();
  test<int*>();
  return true;
}

int main(int, char**) {
  tests();
#if TEST_STD_VER > 17 && defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(tests(), "");
#endif // TEST_STD_VER > 17 && defined(_LIBCUDACXX_ADDRESSOF)

  return 0;
}
