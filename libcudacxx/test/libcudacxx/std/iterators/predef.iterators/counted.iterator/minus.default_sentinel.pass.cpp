//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// friend constexpr iter_difference_t<I> operator-(
//   const counted_iterator& x, default_sentinel_t);
// friend constexpr iter_difference_t<I> operator-(
//   default_sentinel_t, const counted_iterator& y);

#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    cuda::std::counted_iterator iter(random_access_iterator<int*>{buffer}, 8);
    assert(iter - cuda::std::default_sentinel == -8);
    assert(cuda::std::default_sentinel - iter == 8);
    assert(iter.count() == 8);

    ASSERT_SAME_TYPE(decltype(iter - cuda::std::default_sentinel), cuda::std::iter_difference_t<int*>);
    ASSERT_SAME_TYPE(decltype(cuda::std::default_sentinel - iter), cuda::std::iter_difference_t<int*>);
  }
  {
    const cuda::std::counted_iterator iter(random_access_iterator<int*>{buffer}, 8);
    assert(iter - cuda::std::default_sentinel == -8);
    assert(cuda::std::default_sentinel - iter == 8);
    assert(iter.count() == 8);

    ASSERT_SAME_TYPE(decltype(iter - cuda::std::default_sentinel), cuda::std::iter_difference_t<int*>);
    ASSERT_SAME_TYPE(decltype(cuda::std::default_sentinel - iter), cuda::std::iter_difference_t<int*>);
  }
  {
    cuda::std::counted_iterator iter(forward_iterator<int*>{buffer}, 8);
    assert(iter - cuda::std::default_sentinel == -8);
    assert(cuda::std::default_sentinel - iter == 8);
    assert(iter.count() == 8);

    ASSERT_SAME_TYPE(decltype(iter - cuda::std::default_sentinel), cuda::std::iter_difference_t<int*>);
    ASSERT_SAME_TYPE(decltype(cuda::std::default_sentinel - iter), cuda::std::iter_difference_t<int*>);
  }
  {
    const cuda::std::counted_iterator iter(forward_iterator<int*>{buffer}, 8);
    assert(iter - cuda::std::default_sentinel == -8);
    assert(cuda::std::default_sentinel - iter == 8);
    assert(iter.count() == 8);

    ASSERT_SAME_TYPE(decltype(iter - cuda::std::default_sentinel), cuda::std::iter_difference_t<int*>);
    ASSERT_SAME_TYPE(decltype(cuda::std::default_sentinel - iter), cuda::std::iter_difference_t<int*>);
  }
  {
    cuda::std::counted_iterator iter(cpp20_input_iterator<int*>{buffer}, 8);
    assert(iter - cuda::std::default_sentinel == -8);
    assert(cuda::std::default_sentinel - iter == 8);
    assert(iter.count() == 8);

    ASSERT_SAME_TYPE(decltype(iter - cuda::std::default_sentinel), cuda::std::iter_difference_t<int*>);
    ASSERT_SAME_TYPE(decltype(cuda::std::default_sentinel - iter), cuda::std::iter_difference_t<int*>);
  }
  {
    const cuda::std::counted_iterator iter(cpp20_input_iterator<int*>{buffer}, 8);
    assert(iter - cuda::std::default_sentinel == -8);
    assert(cuda::std::default_sentinel - iter == 8);
    assert(iter.count() == 8);

    ASSERT_SAME_TYPE(decltype(iter - cuda::std::default_sentinel), cuda::std::iter_difference_t<int*>);
    ASSERT_SAME_TYPE(decltype(cuda::std::default_sentinel - iter), cuda::std::iter_difference_t<int*>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
