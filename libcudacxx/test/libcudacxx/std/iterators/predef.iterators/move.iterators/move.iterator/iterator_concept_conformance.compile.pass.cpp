//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// move_iterator

#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ void test()
{
  {
    using iterator = cuda::std::move_iterator<cpp17_input_iterator<int*>>;

    static_assert(!cuda::std::default_initializable<iterator>);
    static_assert(cuda::std::copyable<iterator>);
    static_assert(cuda::std::input_iterator<iterator>);
    static_assert(!cuda::std::forward_iterator<iterator>);
    static_assert(!cuda::std::sentinel_for<iterator, iterator>); // not copyable
    static_assert(!cuda::std::sized_sentinel_for<iterator, iterator>);
    static_assert(!cuda::std::indirectly_movable<int*, iterator>);
    static_assert(!cuda::std::indirectly_movable_storable<int*, iterator>);
    static_assert(!cuda::std::indirectly_copyable<int*, iterator>);
    static_assert(!cuda::std::indirectly_copyable_storable<int*, iterator>);
    static_assert(cuda::std::indirectly_readable<iterator>);
    static_assert(!cuda::std::indirectly_writable<iterator, int>);
#if !defined(TEST_COMPILER_MSVC_2017)
    static_assert(cuda::std::indirectly_swappable<iterator, iterator>);
#endif // !TEST_COMPILER_MSVC_2017
  }
  {
    using iterator = cuda::std::move_iterator<cpp20_input_iterator<int*>>;

    static_assert(!cuda::std::default_initializable<iterator>);
#if !defined(TEST_COMPILER_MSVC_2017)
    static_assert(!cuda::std::copyable<iterator>);
#endif // !TEST_COMPILER_MSVC_2017
    static_assert(cuda::std::input_iterator<iterator>);
    static_assert(!cuda::std::forward_iterator<iterator>);
    static_assert(!cuda::std::sentinel_for<iterator, iterator>); // not copyable
    static_assert(!cuda::std::sized_sentinel_for<iterator, iterator>);
    static_assert(!cuda::std::indirectly_movable<int*, iterator>);
    static_assert(!cuda::std::indirectly_movable_storable<int*, iterator>);
    static_assert(!cuda::std::indirectly_copyable<int*, iterator>);
    static_assert(!cuda::std::indirectly_copyable_storable<int*, iterator>);
    static_assert(cuda::std::indirectly_readable<iterator>);
    static_assert(!cuda::std::indirectly_writable<iterator, int>);
#if !defined(TEST_COMPILER_MSVC_2017)
    static_assert(cuda::std::indirectly_swappable<iterator, iterator>);
#endif // !TEST_COMPILER_MSVC_2017
  }
  {
    using iterator = cuda::std::move_iterator<forward_iterator<int*>>;

    static_assert(cuda::std::default_initializable<iterator>);
    static_assert(cuda::std::copyable<iterator>);
    static_assert(cuda::std::forward_iterator<iterator>);
    static_assert(!cuda::std::bidirectional_iterator<iterator>);
    static_assert(cuda::std::sentinel_for<iterator, iterator>);
    static_assert(!cuda::std::sized_sentinel_for<iterator, iterator>);
    static_assert(!cuda::std::indirectly_movable<int*, iterator>);
    static_assert(!cuda::std::indirectly_movable_storable<int*, iterator>);
    static_assert(!cuda::std::indirectly_copyable<int*, iterator>);
    static_assert(!cuda::std::indirectly_copyable_storable<int*, iterator>);
    static_assert(cuda::std::indirectly_readable<iterator>);
    static_assert(!cuda::std::indirectly_writable<iterator, int>);
#if !defined(TEST_COMPILER_MSVC_2017)
    static_assert(cuda::std::indirectly_swappable<iterator, iterator>);
#endif // !TEST_COMPILER_MSVC_2017
  }
  {
    using iterator = cuda::std::move_iterator<bidirectional_iterator<int*>>;

    static_assert(cuda::std::default_initializable<iterator>);
    static_assert(cuda::std::copyable<iterator>);
    static_assert(cuda::std::bidirectional_iterator<iterator>);
    static_assert(!cuda::std::random_access_iterator<iterator>);
    static_assert(cuda::std::sentinel_for<iterator, iterator>);
    static_assert(!cuda::std::sized_sentinel_for<iterator, iterator>);
    static_assert(!cuda::std::indirectly_movable<int*, iterator>);
    static_assert(!cuda::std::indirectly_movable_storable<int*, iterator>);
    static_assert(!cuda::std::indirectly_copyable<int*, iterator>);
    static_assert(!cuda::std::indirectly_copyable_storable<int*, iterator>);
    static_assert(cuda::std::indirectly_readable<iterator>);
    static_assert(!cuda::std::indirectly_writable<iterator, int>);
#if !defined(TEST_COMPILER_MSVC_2017)
    static_assert(cuda::std::indirectly_swappable<iterator, iterator>);
#endif // !TEST_COMPILER_MSVC_2017
  }
  {
    using iterator = cuda::std::move_iterator<random_access_iterator<int*>>;

    static_assert(cuda::std::default_initializable<iterator>);
    static_assert(cuda::std::copyable<iterator>);
    static_assert(cuda::std::random_access_iterator<iterator>);
    static_assert(!cuda::std::contiguous_iterator<iterator>);
    static_assert(cuda::std::sentinel_for<iterator, iterator>);
    static_assert(cuda::std::sized_sentinel_for<iterator, iterator>);
    static_assert(!cuda::std::indirectly_movable<int*, iterator>);
    static_assert(!cuda::std::indirectly_movable_storable<int*, iterator>);
    static_assert(!cuda::std::indirectly_copyable<int*, iterator>);
    static_assert(!cuda::std::indirectly_copyable_storable<int*, iterator>);
    static_assert(cuda::std::indirectly_readable<iterator>);
    static_assert(!cuda::std::indirectly_writable<iterator, int>);
#if !defined(TEST_COMPILER_MSVC_2017)
    static_assert(cuda::std::indirectly_swappable<iterator, iterator>);
#endif // !TEST_COMPILER_MSVC_2017
  }
  {
    using iterator = cuda::std::move_iterator<contiguous_iterator<int*>>;

    static_assert(cuda::std::default_initializable<iterator>);
    static_assert(cuda::std::copyable<iterator>);
    static_assert(cuda::std::random_access_iterator<iterator>);
    static_assert(!cuda::std::contiguous_iterator<iterator>);
    static_assert(cuda::std::sentinel_for<iterator, iterator>);
    static_assert(cuda::std::sized_sentinel_for<iterator, iterator>);
    static_assert(!cuda::std::indirectly_movable<int*, iterator>);
    static_assert(!cuda::std::indirectly_movable_storable<int*, iterator>);
    static_assert(!cuda::std::indirectly_copyable<int*, iterator>);
    static_assert(!cuda::std::indirectly_copyable_storable<int*, iterator>);
    static_assert(cuda::std::indirectly_readable<iterator>);
    static_assert(!cuda::std::indirectly_writable<iterator, int>);
#if !defined(TEST_COMPILER_MSVC_2017)
    static_assert(cuda::std::indirectly_swappable<iterator, iterator>);
#endif // !TEST_COMPILER_MSVC_2017
  }
  {
    using iterator = cuda::std::move_iterator<int*>;

    static_assert(cuda::std::default_initializable<iterator>);
    static_assert(cuda::std::copyable<iterator>);
    static_assert(cuda::std::random_access_iterator<iterator>);
    static_assert(!cuda::std::contiguous_iterator<iterator>);
    static_assert(cuda::std::sentinel_for<iterator, iterator>);
    static_assert(cuda::std::sized_sentinel_for<iterator, iterator>);
    static_assert(!cuda::std::indirectly_movable<int*, iterator>);
    static_assert(!cuda::std::indirectly_movable_storable<int*, iterator>);
    static_assert(!cuda::std::indirectly_copyable<int*, iterator>);
    static_assert(!cuda::std::indirectly_copyable_storable<int*, iterator>);
    static_assert(cuda::std::indirectly_readable<iterator>);
    static_assert(!cuda::std::indirectly_writable<iterator, int>);
#if !defined(TEST_COMPILER_MSVC_2017)
    static_assert(cuda::std::indirectly_swappable<iterator, iterator>);
#endif // !TEST_COMPILER_MSVC_2017
  }
}

int main(int, char**)
{
  return 0;
}
