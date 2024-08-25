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

// iterator() requires default_initializable<OuterIter> = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>

#include "../types.h"
#include "test_iterators.h"

template <class It>
struct view : cuda::std::ranges::view_base
{
  __host__ __device__ It begin() const
  {
    return It{nullptr};
  }
  __host__ __device__ sentinel_wrapper<It> end() const
  {
    return sentinel_wrapper<It>{};
  }
};

template <class It>
__host__ __device__ constexpr void test_default_constructible()
{
  using JoinView     = cuda::std::ranges::join_view<view<It>>;
  using JoinIterator = cuda::std::ranges::iterator_t<JoinView>;
  static_assert(cuda::std::is_default_constructible_v<JoinIterator>);
  JoinIterator it;
  (void) it;
}

__host__ __device__ constexpr bool test()
{
  test_default_constructible<cpp17_input_iterator<ChildView*>>();
  test_default_constructible<cpp20_input_iterator<ChildView*>>();
  test_default_constructible<forward_iterator<ChildView*>>();
  test_default_constructible<bidirectional_iterator<ChildView*>>();
  test_default_constructible<random_access_iterator<ChildView*>>();
  test_default_constructible<contiguous_iterator<ChildView*>>();
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
