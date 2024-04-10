//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// template<class I>
//   concept permutable = see below; // Since C++20

#include <cuda/std/iterator>

#include "test_iterators.h"

using AllConstraintsSatisfied = forward_iterator<int*>;
static_assert(cuda::std::forward_iterator<AllConstraintsSatisfied>);
static_assert(cuda::std::indirectly_movable_storable<AllConstraintsSatisfied, AllConstraintsSatisfied>);
static_assert(cuda::std::indirectly_swappable<AllConstraintsSatisfied>);
static_assert(cuda::std::permutable<AllConstraintsSatisfied>);

using NotAForwardIterator = cpp20_input_iterator<int*>;
static_assert(!cuda::std::forward_iterator<NotAForwardIterator>);
static_assert(cuda::std::indirectly_movable_storable<NotAForwardIterator, NotAForwardIterator>);
static_assert(cuda::std::indirectly_swappable<NotAForwardIterator>);
static_assert(!cuda::std::permutable<NotAForwardIterator>);

#if !defined(TEST_COMPILER_CUDACC_BELOW_11_3) && !defined(TEST_COMPILER_MSVC_2017)
struct NonCopyable
{
  NonCopyable(const NonCopyable&)            = delete;
  NonCopyable& operator=(const NonCopyable&) = delete;
  __host__ __device__ friend void swap(NonCopyable&, NonCopyable&);
};
using NotIMS = forward_iterator<NonCopyable*>;

static_assert(cuda::std::forward_iterator<NotIMS>);
static_assert(!cuda::std::indirectly_movable_storable<NotIMS, NotIMS>);
static_assert(cuda::std::indirectly_swappable<NotIMS>);
static_assert(!cuda::std::permutable<NotIMS>);
#endif // !TEST_COMPILER_CUDACC_BELOW_11_3 && !TEST_COMPILER_MSVC_2017

// Note: it is impossible for an iterator to satisfy `indirectly_movable_storable` but not `indirectly_swappable`:
// `indirectly_swappable` requires both iterators to be `indirectly_readable` and for `ranges::iter_swap` to be
// well-formed for both iterators. `indirectly_movable_storable` also requires the iterator to be `indirectly_readable`.
// `ranges::iter_swap` is always defined for `indirectly_movable_storable` iterators.

int main(int, char**)
{
  return 0;
}
