//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>
//
// reverse_iterator
//
// friend constexpr iter_rvalue_reference_t<Iterator>
//   iter_move(const reverse_iterator& i) noexcept(see below);

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "test_macros.h"

TEST_FUNC constexpr bool test()
{
  // Can use `iter_move` with a regular array.
  {
    constexpr int N = 3;
    int a[N]        = {0, 1, 2};

    cuda::std::reverse_iterator<int*> ri(a + N);
    static_assert(cuda::std::same_as<decltype(iter_move(ri)), int&&>);
    assert(iter_move(ri) == 2);

    ++ri;
    assert(iter_move(ri) == 1);
  }

  // Check that the `iter_move` customization point is being used.
  {
    constexpr int N = 3;
    int a[N]        = {0, 1, 2};

    int iter_move_invocations = 0;
    adl::Iterator i           = adl::Iterator::TrackMoves(a + N, iter_move_invocations);
    cuda::std::reverse_iterator<adl::Iterator> ri(i);
    int x = iter_move(ri);
    assert(x == 2);
    assert(iter_move_invocations == 1);
  }

  // Check the `noexcept` specification.
#if !TEST_COMPILER(GCC, <, 8) // ancient gcc trips over this not being a literal type
  {
    {
      struct ThrowingCopyNoexceptDecrement
      {
        using value_type      = int;
        using difference_type = ptrdiff_t;

        TEST_FUNC ThrowingCopyNoexceptDecrement();
        TEST_FUNC ThrowingCopyNoexceptDecrement(const ThrowingCopyNoexceptDecrement&);

        TEST_FUNC int& operator*() const noexcept
        {
          static int x;
          return x;
        }

        TEST_FUNC ThrowingCopyNoexceptDecrement& operator++();
        TEST_FUNC ThrowingCopyNoexceptDecrement operator++(int);
        TEST_FUNC ThrowingCopyNoexceptDecrement& operator--() noexcept;
        TEST_FUNC ThrowingCopyNoexceptDecrement operator--(int) noexcept;
#  if TEST_STD_VER > 2017
        bool operator==(const ThrowingCopyNoexceptDecrement&) const = default;
#  else
        TEST_FUNC bool operator==(const ThrowingCopyNoexceptDecrement&) const;
        TEST_FUNC bool operator!=(const ThrowingCopyNoexceptDecrement&) const;
#  endif
      };
      static_assert(cuda::std::bidirectional_iterator<ThrowingCopyNoexceptDecrement>);

      static_assert(!cuda::std::is_nothrow_copy_constructible_v<ThrowingCopyNoexceptDecrement>);
      static_assert(noexcept(cuda::std::ranges::iter_move(--cuda::std::declval<ThrowingCopyNoexceptDecrement&>())));
      using RI = cuda::std::reverse_iterator<ThrowingCopyNoexceptDecrement>;
      static_assert(!noexcept(iter_move(cuda::std::declval<RI>())));
    }

    {
      struct NoexceptCopyThrowingDecrement
      {
        using value_type      = int;
        using difference_type = ptrdiff_t;

        TEST_FUNC NoexceptCopyThrowingDecrement();
        TEST_FUNC NoexceptCopyThrowingDecrement(const NoexceptCopyThrowingDecrement&) noexcept;

        TEST_FUNC int& operator*() const
        {
          static int x;
          return x;
        }

        TEST_FUNC NoexceptCopyThrowingDecrement& operator++();
        TEST_FUNC NoexceptCopyThrowingDecrement operator++(int);
        TEST_FUNC NoexceptCopyThrowingDecrement& operator--();
        TEST_FUNC NoexceptCopyThrowingDecrement operator--(int);

#  if TEST_STD_VER > 2017
        bool operator==(const NoexceptCopyThrowingDecrement&) const = default;
#  else
        TEST_FUNC bool operator==(const NoexceptCopyThrowingDecrement&) const;
        TEST_FUNC bool operator!=(const NoexceptCopyThrowingDecrement&) const;
#  endif
      };
      static_assert(cuda::std::bidirectional_iterator<NoexceptCopyThrowingDecrement>);

      static_assert(cuda::std::is_nothrow_copy_constructible_v<NoexceptCopyThrowingDecrement>);
      static_assert(!noexcept(cuda::std::ranges::iter_move(--cuda::std::declval<NoexceptCopyThrowingDecrement&>())));
      using RI = cuda::std::reverse_iterator<NoexceptCopyThrowingDecrement>;
      static_assert(!noexcept(iter_move(cuda::std::declval<RI>())));
    }

    {
      struct NoexceptCopyAndDecrement
      {
        using value_type      = int;
        using difference_type = ptrdiff_t;

        TEST_FUNC NoexceptCopyAndDecrement();
        TEST_FUNC NoexceptCopyAndDecrement(const NoexceptCopyAndDecrement&) noexcept;

        TEST_FUNC int& operator*() const noexcept
        {
          static int x;
          return x;
        }

        TEST_FUNC NoexceptCopyAndDecrement& operator++();
        TEST_FUNC NoexceptCopyAndDecrement operator++(int);
        TEST_FUNC NoexceptCopyAndDecrement& operator--() noexcept;
        TEST_FUNC NoexceptCopyAndDecrement operator--(int) noexcept;

#  if TEST_STD_VER > 2017
        bool operator==(const NoexceptCopyAndDecrement&) const = default;
#  else
        TEST_FUNC bool operator==(const NoexceptCopyAndDecrement&) const;
        TEST_FUNC bool operator!=(const NoexceptCopyAndDecrement&) const;
#  endif
      };
      static_assert(cuda::std::bidirectional_iterator<NoexceptCopyAndDecrement>);

      static_assert(cuda::std::is_nothrow_copy_constructible_v<NoexceptCopyAndDecrement>);
      static_assert(noexcept(cuda::std::ranges::iter_move(--cuda::std::declval<NoexceptCopyAndDecrement&>())));
      using RI = cuda::std::reverse_iterator<NoexceptCopyAndDecrement>;
      static_assert(noexcept(iter_move(cuda::std::declval<RI>())));
    }
  }
#endif // !TEST_COMPILER(GCC, <, 8)

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
