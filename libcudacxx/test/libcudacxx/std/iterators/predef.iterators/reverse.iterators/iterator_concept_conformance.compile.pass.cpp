//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// reverse_iterator

#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class I1>
__host__ __device__ constexpr bool common_reverse_iterator_checks()
{
  static_assert(cuda::std::indirectly_writable<I1, int>);
  static_assert(cuda::std::sentinel_for<I1, I1>);
  return true;
}

using reverse_bidirectional_iterator = cuda::std::reverse_iterator<bidirectional_iterator<int*>>;
static_assert(common_reverse_iterator_checks<reverse_bidirectional_iterator>());
static_assert(cuda::std::bidirectional_iterator<reverse_bidirectional_iterator>);
static_assert(!cuda::std::random_access_iterator<reverse_bidirectional_iterator>);
static_assert(!cuda::std::sized_sentinel_for<reverse_bidirectional_iterator, reverse_bidirectional_iterator>);
static_assert(cuda::std::indirectly_movable<reverse_bidirectional_iterator, reverse_bidirectional_iterator>);
static_assert(cuda::std::indirectly_movable_storable<reverse_bidirectional_iterator, reverse_bidirectional_iterator>);
static_assert(cuda::std::indirectly_copyable<reverse_bidirectional_iterator, reverse_bidirectional_iterator>);
static_assert(cuda::std::indirectly_copyable_storable<reverse_bidirectional_iterator, reverse_bidirectional_iterator>);
static_assert(cuda::std::indirectly_swappable<reverse_bidirectional_iterator, reverse_bidirectional_iterator>);

using reverse_random_access_iterator = cuda::std::reverse_iterator<random_access_iterator<int*>>;
static_assert(common_reverse_iterator_checks<reverse_random_access_iterator>());
static_assert(cuda::std::random_access_iterator<reverse_random_access_iterator>);
#if TEST_STD_VER >= 2020 // There is no contiguous_iterator_tag in C++17 so we should not require it
static_assert(!cuda::std::contiguous_iterator<reverse_random_access_iterator>);
#endif // TEST_STD_VER >= 2020
static_assert(cuda::std::sized_sentinel_for<reverse_random_access_iterator, reverse_random_access_iterator>);
static_assert(cuda::std::indirectly_movable<reverse_random_access_iterator, reverse_random_access_iterator>);
static_assert(cuda::std::indirectly_movable_storable<reverse_random_access_iterator, reverse_random_access_iterator>);
static_assert(cuda::std::indirectly_copyable<reverse_random_access_iterator, reverse_random_access_iterator>);
static_assert(cuda::std::indirectly_copyable_storable<reverse_random_access_iterator, reverse_random_access_iterator>);
static_assert(cuda::std::indirectly_swappable<reverse_random_access_iterator, reverse_random_access_iterator>);

using reverse_contiguous_iterator = cuda::std::reverse_iterator<contiguous_iterator<int*>>;
static_assert(common_reverse_iterator_checks<reverse_contiguous_iterator>());
static_assert(cuda::std::random_access_iterator<reverse_contiguous_iterator>);
#if TEST_STD_VER >= 2020 // There is no contiguous_iterator_tag in C++17 so we should not require it
static_assert(!cuda::std::contiguous_iterator<reverse_contiguous_iterator>);
#endif // TEST_STD_VER >= 2020
static_assert(cuda::std::sized_sentinel_for<reverse_contiguous_iterator, reverse_contiguous_iterator>);
static_assert(cuda::std::indirectly_movable<reverse_contiguous_iterator, reverse_contiguous_iterator>);
static_assert(cuda::std::indirectly_movable_storable<reverse_contiguous_iterator, reverse_contiguous_iterator>);
static_assert(cuda::std::indirectly_copyable<reverse_contiguous_iterator, reverse_contiguous_iterator>);
static_assert(cuda::std::indirectly_copyable_storable<reverse_contiguous_iterator, reverse_contiguous_iterator>);
static_assert(cuda::std::indirectly_swappable<reverse_contiguous_iterator, reverse_contiguous_iterator>);

static_assert(
  cuda::std::equality_comparable_with<cuda::std::reverse_iterator<int*>, cuda::std::reverse_iterator<const int*>>);
static_assert(
  !cuda::std::equality_comparable_with<cuda::std::reverse_iterator<int*>, cuda::std::reverse_iterator<char*>>);
static_assert(
  cuda::std::totally_ordered_with<cuda::std::reverse_iterator<int*>, cuda::std::reverse_iterator<const int*>>);
static_assert(!cuda::std::totally_ordered_with<cuda::std::reverse_iterator<int*>, cuda::std::reverse_iterator<char*>>);
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
static_assert(
  cuda::std::three_way_comparable_with<cuda::std::reverse_iterator<int*>, cuda::std::reverse_iterator<const int*>>);
static_assert(
  !cuda::std::three_way_comparable_with<cuda::std::reverse_iterator<int*>, cuda::std::reverse_iterator<char*>>);
#endif // TEST_HAS_NO_SPACESHIP_OPERATOR

int main(int, char**)
{
  return 0;
}
