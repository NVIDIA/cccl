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
// template<indirectly_swappable<Iterator> Iterator2>
//   friend constexpr void
//     iter_swap(const move_iterator& x, const move_iterator<Iterator2>& y)
//       noexcept(noexcept(ranges::iter_swap(x.current, y.current))); // Since C++20

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "test_macros.h"

template <bool IsNoexcept>
struct MaybeNoexceptSwap
{
  using value_type      = int;
  using difference_type = ptrdiff_t;

  __host__ __device__ constexpr friend void iter_swap(MaybeNoexceptSwap, MaybeNoexceptSwap) noexcept(IsNoexcept) {}

  __host__ __device__ int& operator*() const
  {
    static int x;
    return x;
  }

  __host__ __device__ MaybeNoexceptSwap& operator++();
  __host__ __device__ MaybeNoexceptSwap operator++(int);
};
using ThrowingBase = MaybeNoexceptSwap<false>;
using NoexceptBase = MaybeNoexceptSwap<true>;
static_assert(cuda::std::input_iterator<ThrowingBase>);
#if !defined(TEST_COMPILER_ICC)
ASSERT_NOT_NOEXCEPT(
  cuda::std::ranges::iter_swap(cuda::std::declval<ThrowingBase>(), cuda::std::declval<ThrowingBase>()));
#  if !defined(TEST_COMPILER_MSVC_2017) // MSVC2017 gets confused by the two friends and only considers the first
ASSERT_NOEXCEPT(cuda::std::ranges::iter_swap(cuda::std::declval<NoexceptBase>(), cuda::std::declval<NoexceptBase>()));
#  endif // !TEST_COMPILER_MSVC_2017
#endif // & !TEST_COMPILER_ICC

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  // Can use `iter_swap` with a regular array.
  {
    int a[] = {0, 1, 2};

    cuda::std::move_iterator<int*> b(a);
    cuda::std::move_iterator<int*> e(a + 2);
    assert(a[0] == 0);
    assert(a[2] == 2);

    static_assert(cuda::std::same_as<decltype(iter_swap(b, e)), void>);
    iter_swap(b, e);
    assert(a[0] == 2);
    assert(a[2] == 0);
  }

  // Check that the `iter_swap` customization point is being used.
  {
    int iter_swap_invocations = 0;
    adl::Iterator base1       = adl::Iterator::TrackSwaps(iter_swap_invocations);
    adl::Iterator base2       = adl::Iterator::TrackSwaps(iter_swap_invocations);
    cuda::std::move_iterator<adl::Iterator> i1(base1), i2(base2);
    iter_swap(i1, i2);
    assert(iter_swap_invocations == 1);

    iter_swap(i2, i1);
    assert(iter_swap_invocations == 2);
  }

  // Check the `noexcept` specification.
  {
#if !defined(TEST_COMPILER_ICC)
    using ThrowingIter = cuda::std::move_iterator<ThrowingBase>;
    ASSERT_NOT_NOEXCEPT(iter_swap(cuda::std::declval<ThrowingIter>(), cuda::std::declval<ThrowingIter>()));
#  if !defined(TEST_COMPILER_MSVC_2017) // MSVC2017 gets confused by the two friends and only considers the first
    using NoexceptIter = cuda::std::move_iterator<NoexceptBase>;
    ASSERT_NOEXCEPT(iter_swap(cuda::std::declval<NoexceptIter>(), cuda::std::declval<NoexceptIter>()));
#  endif // !TEST_COMPILER_MSVC_2017
#endif // !TEST_COMPILER_ICC
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017
  static_assert(test());
#endif

  return 0;
}
