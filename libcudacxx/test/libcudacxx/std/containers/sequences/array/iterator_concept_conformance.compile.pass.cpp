//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// iterator, const_iterator, reverse_iterator, const_reverse_iterator

#include <cuda/std/array>
#include <cuda/std/iterator>

using iterator               = cuda::std::array<int, 10>::iterator;
using const_iterator         = cuda::std::array<int, 10>::const_iterator;
using reverse_iterator       = cuda::std::array<int, 10>::reverse_iterator;
using const_reverse_iterator = cuda::std::array<int, 10>::const_reverse_iterator;

static_assert(cuda::std::contiguous_iterator<iterator>);
static_assert(cuda::std::indirectly_writable<iterator, int>);
static_assert(cuda::std::sentinel_for<iterator, iterator>);
static_assert(cuda::std::sentinel_for<iterator, const_iterator>);
static_assert(!cuda::std::sentinel_for<iterator, reverse_iterator>);
static_assert(!cuda::std::sentinel_for<iterator, const_reverse_iterator>);
static_assert(cuda::std::sized_sentinel_for<iterator, iterator>);
static_assert(cuda::std::sized_sentinel_for<iterator, const_iterator>);
static_assert(!cuda::std::sized_sentinel_for<iterator, reverse_iterator>);
static_assert(!cuda::std::sized_sentinel_for<iterator, const_reverse_iterator>);
static_assert(cuda::std::indirectly_movable<iterator, iterator>);
static_assert(cuda::std::indirectly_movable_storable<iterator, iterator>);
static_assert(!cuda::std::indirectly_movable<iterator, const_iterator>);
static_assert(!cuda::std::indirectly_movable_storable<iterator, const_iterator>);
static_assert(cuda::std::indirectly_movable<iterator, reverse_iterator>);
static_assert(cuda::std::indirectly_movable_storable<iterator, reverse_iterator>);
static_assert(!cuda::std::indirectly_movable<iterator, const_reverse_iterator>);
static_assert(!cuda::std::indirectly_movable_storable<iterator, const_reverse_iterator>);
static_assert(cuda::std::indirectly_copyable<iterator, iterator>);
static_assert(cuda::std::indirectly_copyable_storable<iterator, iterator>);
static_assert(!cuda::std::indirectly_copyable<iterator, const_iterator>);
static_assert(!cuda::std::indirectly_copyable_storable<iterator, const_iterator>);
static_assert(cuda::std::indirectly_copyable<iterator, reverse_iterator>);
static_assert(cuda::std::indirectly_copyable_storable<iterator, reverse_iterator>);
static_assert(!cuda::std::indirectly_copyable<iterator, const_reverse_iterator>);
static_assert(!cuda::std::indirectly_copyable_storable<iterator, const_reverse_iterator>);
static_assert(cuda::std::indirectly_swappable<iterator, iterator>);

static_assert(cuda::std::contiguous_iterator<const_iterator>);
static_assert(!cuda::std::indirectly_writable<const_iterator, int>);
static_assert(cuda::std::sentinel_for<const_iterator, iterator>);
static_assert(cuda::std::sentinel_for<const_iterator, const_iterator>);
static_assert(!cuda::std::sentinel_for<const_iterator, reverse_iterator>);
static_assert(!cuda::std::sentinel_for<const_iterator, const_reverse_iterator>);
static_assert(cuda::std::sized_sentinel_for<const_iterator, iterator>);
static_assert(cuda::std::sized_sentinel_for<const_iterator, const_iterator>);
static_assert(!cuda::std::sized_sentinel_for<const_iterator, reverse_iterator>);
static_assert(!cuda::std::sized_sentinel_for<const_iterator, const_reverse_iterator>);
static_assert(cuda::std::indirectly_movable<const_iterator, iterator>);
static_assert(cuda::std::indirectly_movable_storable<const_iterator, iterator>);
static_assert(!cuda::std::indirectly_movable<const_iterator, const_iterator>);
static_assert(!cuda::std::indirectly_movable_storable<const_iterator, const_iterator>);
static_assert(cuda::std::indirectly_movable<const_iterator, reverse_iterator>);
static_assert(cuda::std::indirectly_movable_storable<const_iterator, reverse_iterator>);
static_assert(!cuda::std::indirectly_movable<const_iterator, const_reverse_iterator>);
static_assert(!cuda::std::indirectly_movable_storable<const_iterator, const_reverse_iterator>);
static_assert(cuda::std::indirectly_copyable<const_iterator, iterator>);
static_assert(cuda::std::indirectly_copyable_storable<const_iterator, iterator>);
static_assert(!cuda::std::indirectly_copyable<const_iterator, const_iterator>);
static_assert(!cuda::std::indirectly_copyable_storable<const_iterator, const_iterator>);
static_assert(cuda::std::indirectly_copyable<const_iterator, reverse_iterator>);
static_assert(cuda::std::indirectly_copyable_storable<const_iterator, reverse_iterator>);
static_assert(!cuda::std::indirectly_copyable<const_iterator, const_reverse_iterator>);
static_assert(!cuda::std::indirectly_copyable_storable<const_iterator, const_reverse_iterator>);
static_assert(!cuda::std::indirectly_swappable<const_iterator, const_iterator>);

int main(int, char**)
{
  return 0;
}
