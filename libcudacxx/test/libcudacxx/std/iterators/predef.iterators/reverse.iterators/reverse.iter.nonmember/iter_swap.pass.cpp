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
// reverse_iterator
//
// template<indirectly_swappable<Iterator> Iterator2>
//   friend constexpr void
//     iter_swap(const reverse_iterator& x,
//               const reverse_iterator<Iterator2>& y) noexcept(see below);

#ifdef __clang__
#pragma clang diagnostic ignored "-Wunevaluated-expression"
#endif

#include <cuda/std/iterator>

#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include "test_iterators.h"
#include "test_macros.h"

struct ThrowingCopyNoexceptDecrement {
  using value_type = int;
  using difference_type = ptrdiff_t;

  TEST_HOST_DEVICE ThrowingCopyNoexceptDecrement();
  TEST_HOST_DEVICE ThrowingCopyNoexceptDecrement(const ThrowingCopyNoexceptDecrement&);

  TEST_HOST_DEVICE int& operator*() const noexcept { static int x; return x; }

  TEST_HOST_DEVICE ThrowingCopyNoexceptDecrement& operator++();
  TEST_HOST_DEVICE ThrowingCopyNoexceptDecrement operator++(int);
  TEST_HOST_DEVICE ThrowingCopyNoexceptDecrement& operator--() noexcept;
  TEST_HOST_DEVICE ThrowingCopyNoexceptDecrement operator--(int) noexcept;

#if TEST_STD_VER > 2017
  bool operator==(const ThrowingCopyNoexceptDecrement&) const = default;
#else
  TEST_HOST_DEVICE bool operator==(const ThrowingCopyNoexceptDecrement&) const;
  TEST_HOST_DEVICE bool operator!=(const ThrowingCopyNoexceptDecrement&) const;
#endif
};

struct NoexceptCopyThrowingDecrement {
  using value_type = int;
  using difference_type = ptrdiff_t;

  TEST_HOST_DEVICE NoexceptCopyThrowingDecrement();
  TEST_HOST_DEVICE NoexceptCopyThrowingDecrement(const NoexceptCopyThrowingDecrement&) noexcept;

  TEST_HOST_DEVICE int& operator*() const { static int x; return x; }

  TEST_HOST_DEVICE NoexceptCopyThrowingDecrement& operator++();
  TEST_HOST_DEVICE NoexceptCopyThrowingDecrement operator++(int);
  TEST_HOST_DEVICE NoexceptCopyThrowingDecrement& operator--();
  TEST_HOST_DEVICE NoexceptCopyThrowingDecrement operator--(int);

#if TEST_STD_VER > 2017
  bool operator==(const NoexceptCopyThrowingDecrement&) const = default;
#else
  TEST_HOST_DEVICE bool operator==(const NoexceptCopyThrowingDecrement&) const;
  TEST_HOST_DEVICE bool operator!=(const NoexceptCopyThrowingDecrement&) const;
#endif
};

struct NoexceptCopyAndDecrement {
  using value_type = int;
  using difference_type = ptrdiff_t;

  TEST_HOST_DEVICE NoexceptCopyAndDecrement();
  TEST_HOST_DEVICE NoexceptCopyAndDecrement(const NoexceptCopyAndDecrement&) noexcept;

  TEST_HOST_DEVICE int& operator*() const noexcept { static int x; return x; }

  TEST_HOST_DEVICE NoexceptCopyAndDecrement& operator++();
  TEST_HOST_DEVICE NoexceptCopyAndDecrement operator++(int);
  TEST_HOST_DEVICE NoexceptCopyAndDecrement& operator--() noexcept;
  TEST_HOST_DEVICE NoexceptCopyAndDecrement operator--(int) noexcept;

#if TEST_STD_VER > 2017
  bool operator==(const NoexceptCopyAndDecrement&) const = default;
#else
  TEST_HOST_DEVICE bool operator==(const NoexceptCopyAndDecrement&) const;
  TEST_HOST_DEVICE bool operator!=(const NoexceptCopyAndDecrement&) const;
#endif
};

TEST_HOST_DEVICE TEST_CONSTEXPR_CXX20 bool test() {
  // Can use `iter_swap` with a regular array.
  {
    constexpr int N = 3;
    int a[N] = {0, 1, 2};

    cuda::std::reverse_iterator<int*> rb(a + N);
    cuda::std::reverse_iterator<int*> re(a + 1);
    assert(a[0] == 0);
    assert(a[2] == 2);

    static_assert(cuda::std::same_as<decltype(iter_swap(rb, re)), void>);
    iter_swap(rb, re);
    assert(a[0] == 2);
    assert(a[2] == 0);
  }

  // Check that the `iter_swap` customization point is being used.
  {
    int iter_swap_invocations = 0;
    int a[] = {0, 1, 2};
    adl::Iterator base1 = adl::Iterator::TrackSwaps(a + 1, iter_swap_invocations);
    adl::Iterator base2 = adl::Iterator::TrackSwaps(a + 2, iter_swap_invocations);
    cuda::std::reverse_iterator<adl::Iterator> ri1(base1), ri2(base2);
    iter_swap(ri1, ri2);
    assert(iter_swap_invocations == 1);

    iter_swap(ri2, ri1);
    assert(iter_swap_invocations == 2);
  }

  // Check the `noexcept` specification.
  {
    {
      static_assert(cuda::std::bidirectional_iterator<ThrowingCopyNoexceptDecrement>);

#ifndef TEST_COMPILER_ICC
      static_assert(!cuda::std::is_nothrow_copy_constructible_v<ThrowingCopyNoexceptDecrement>);
#endif // TEST_COMPILER_ICC
      static_assert( cuda::std::is_nothrow_copy_constructible_v<int*>);
#if TEST_STD_VER > 2017
      ASSERT_NOEXCEPT(cuda::std::ranges::iter_swap(--cuda::std::declval<ThrowingCopyNoexceptDecrement&>(), --cuda::std::declval<int*&>()));
#endif
      using RI1 = cuda::std::reverse_iterator<ThrowingCopyNoexceptDecrement>;
      using RI2 = cuda::std::reverse_iterator<int*>;
#ifndef TEST_COMPILER_ICC
      ASSERT_NOT_NOEXCEPT(iter_swap(cuda::std::declval<RI1>(), cuda::std::declval<RI2>()));
      ASSERT_NOT_NOEXCEPT(iter_swap(cuda::std::declval<RI2>(), cuda::std::declval<RI1>()));
#endif // TEST_COMPILER_ICC
    }

    {
      static_assert(cuda::std::bidirectional_iterator<NoexceptCopyThrowingDecrement>);

      static_assert( cuda::std::is_nothrow_copy_constructible_v<NoexceptCopyThrowingDecrement>);
      static_assert( cuda::std::is_nothrow_copy_constructible_v<int*>);
#if TEST_STD_VER > 2017
      ASSERT_NOT_NOEXCEPT(cuda::std::ranges::iter_swap(--cuda::std::declval<NoexceptCopyThrowingDecrement&>(), --cuda::std::declval<int*&>()));
#endif
      using RI1 = cuda::std::reverse_iterator<NoexceptCopyThrowingDecrement>;
      using RI2 = cuda::std::reverse_iterator<int*>;
#ifndef TEST_COMPILER_ICC
      ASSERT_NOT_NOEXCEPT(iter_swap(cuda::std::declval<RI1>(), cuda::std::declval<RI2>()));
      ASSERT_NOT_NOEXCEPT(iter_swap(cuda::std::declval<RI2>(), cuda::std::declval<RI1>()));
#endif // TEST_COMPILER_ICC
    }

    {
      static_assert(cuda::std::bidirectional_iterator<NoexceptCopyAndDecrement>);

      static_assert( cuda::std::is_nothrow_copy_constructible_v<NoexceptCopyAndDecrement>);
      static_assert( cuda::std::is_nothrow_copy_constructible_v<int*>);
#if TEST_STD_VER > 2017
      ASSERT_NOEXCEPT(cuda::std::ranges::iter_swap(--cuda::std::declval<NoexceptCopyAndDecrement&>(), --cuda::std::declval<int*&>()));
#endif
      using RI1 = cuda::std::reverse_iterator<NoexceptCopyAndDecrement>;
      using RI2 = cuda::std::reverse_iterator<int*>;
      ASSERT_NOEXCEPT(iter_swap(cuda::std::declval<RI1>(), cuda::std::declval<RI2>()));
      ASSERT_NOEXCEPT(iter_swap(cuda::std::declval<RI2>(), cuda::std::declval<RI1>()));
    }
  }

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 2017
  static_assert(test());
#endif

  return 0;
}
