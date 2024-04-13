//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <cuda/std/iterator>
//
// friend constexpr iter_rvalue_reference_t<Iterator>
//   iter_move(const move_iterator& i)
//     noexcept(noexcept(ranges::iter_move(i.current))); // Since C++20

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "test_macros.h"

__device__ int global;

template <bool IsNoexcept>
struct MaybeNoexceptMove
{
  int x;
  using value_type      = int;
  using difference_type = ptrdiff_t;

  __host__ __device__ constexpr friend value_type&& iter_move(MaybeNoexceptMove) noexcept(IsNoexcept)
  {
    return cuda::std::move(global);
  }

  __host__ __device__ int& operator*() const
  {
    static int a;
    return a;
  }

  __host__ __device__ MaybeNoexceptMove& operator++();
  __host__ __device__ MaybeNoexceptMove operator++(int);
};
using ThrowingBase = MaybeNoexceptMove<false>;
using NoexceptBase = MaybeNoexceptMove<true>;
static_assert(cuda::std::input_iterator<ThrowingBase>);
#ifndef TEST_COMPILER_ICC
ASSERT_NOT_NOEXCEPT(cuda::std::ranges::iter_move(cuda::std::declval<ThrowingBase>()));
#endif // TEST_COMPILER_ICC
ASSERT_NOEXCEPT(cuda::std::ranges::iter_move(cuda::std::declval<NoexceptBase>()));

__host__ __device__ constexpr bool test()
{
  // Can use `iter_move` with a regular array.
  {
    int a[] = {0, 1, 2};

    cuda::std::move_iterator<int*> i(a);
    static_assert(cuda::std::same_as<decltype(iter_move(i)), int&&>);
    assert(iter_move(i) == 0);

    ++i;
    assert(iter_move(i) == 1);
  }

  // Check that the `iter_move` customization point is being used.
  {
    int a[] = {0, 1, 2};

    int iter_move_invocations = 0;
    adl::Iterator base        = adl::Iterator::TrackMoves(a, iter_move_invocations);
    cuda::std::move_iterator<adl::Iterator> i(base);
    int x = iter_move(i);
    assert(x == 0);
    assert(iter_move_invocations == 1);
  }

  // Check the `noexcept` specification.
  {
#ifndef TEST_COMPILER_ICC
    using ThrowingIter = cuda::std::move_iterator<ThrowingBase>;
    ASSERT_NOT_NOEXCEPT(iter_move(cuda::std::declval<ThrowingIter>()));
#endif // TEST_COMPILER_ICC
    using NoexceptIter = cuda::std::move_iterator<NoexceptBase>;
    ASSERT_NOEXCEPT(iter_move(cuda::std::declval<NoexceptIter>()));
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
