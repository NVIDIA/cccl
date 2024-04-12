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
// friend constexpr iter_rvalue_reference_t<Iterator>
//   iter_move(const reverse_iterator& i) noexcept(see below);

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
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
#if (!defined(TEST_COMPILER_GCC) || __GNUC__ >= 8) // ancient gcc trips over this not being a literal type
  {
    {
      struct ThrowingCopyNoexceptDecrement
      {
        using value_type      = int;
        using difference_type = ptrdiff_t;

        __host__ __device__ ThrowingCopyNoexceptDecrement();
        __host__ __device__ ThrowingCopyNoexceptDecrement(const ThrowingCopyNoexceptDecrement&);

        __host__ __device__ int& operator*() const noexcept
        {
          static int x;
          return x;
        }

        __host__ __device__ ThrowingCopyNoexceptDecrement& operator++();
        __host__ __device__ ThrowingCopyNoexceptDecrement operator++(int);
        __host__ __device__ ThrowingCopyNoexceptDecrement& operator--() noexcept;
        __host__ __device__ ThrowingCopyNoexceptDecrement operator--(int) noexcept;
#  if TEST_STD_VER > 2017
        bool operator==(const ThrowingCopyNoexceptDecrement&) const = default;
#  else
        __host__ __device__ bool operator==(const ThrowingCopyNoexceptDecrement&) const;
        __host__ __device__ bool operator!=(const ThrowingCopyNoexceptDecrement&) const;
#  endif
      };
      static_assert(cuda::std::bidirectional_iterator<ThrowingCopyNoexceptDecrement>);

#  ifndef TEST_COMPILER_ICC
      static_assert(!cuda::std::is_nothrow_copy_constructible_v<ThrowingCopyNoexceptDecrement>);
      ASSERT_NOEXCEPT(cuda::std::ranges::iter_move(--cuda::std::declval<ThrowingCopyNoexceptDecrement&>()));
      using RI = cuda::std::reverse_iterator<ThrowingCopyNoexceptDecrement>;
      ASSERT_NOT_NOEXCEPT(iter_move(cuda::std::declval<RI>()));
#  endif // TEST_COMPILER_ICC
    }

    {
      struct NoexceptCopyThrowingDecrement
      {
        using value_type      = int;
        using difference_type = ptrdiff_t;

        __host__ __device__ NoexceptCopyThrowingDecrement();
        __host__ __device__ NoexceptCopyThrowingDecrement(const NoexceptCopyThrowingDecrement&) noexcept;

        __host__ __device__ int& operator*() const
        {
          static int x;
          return x;
        }

        __host__ __device__ NoexceptCopyThrowingDecrement& operator++();
        __host__ __device__ NoexceptCopyThrowingDecrement operator++(int);
        __host__ __device__ NoexceptCopyThrowingDecrement& operator--();
        __host__ __device__ NoexceptCopyThrowingDecrement operator--(int);

#  if TEST_STD_VER > 2017
        bool operator==(const NoexceptCopyThrowingDecrement&) const = default;
#  else
        __host__ __device__ bool operator==(const NoexceptCopyThrowingDecrement&) const;
        __host__ __device__ bool operator!=(const NoexceptCopyThrowingDecrement&) const;
#  endif
      };
      static_assert(cuda::std::bidirectional_iterator<NoexceptCopyThrowingDecrement>);

      static_assert(cuda::std::is_nothrow_copy_constructible_v<NoexceptCopyThrowingDecrement>);
#  ifndef TEST_COMPILER_ICC
      ASSERT_NOT_NOEXCEPT(cuda::std::ranges::iter_move(--cuda::std::declval<NoexceptCopyThrowingDecrement&>()));
      using RI = cuda::std::reverse_iterator<NoexceptCopyThrowingDecrement>;
      ASSERT_NOT_NOEXCEPT(iter_move(cuda::std::declval<RI>()));
#  endif // TEST_COMPILER_ICC
    }

    {
      struct NoexceptCopyAndDecrement
      {
        using value_type      = int;
        using difference_type = ptrdiff_t;

        __host__ __device__ NoexceptCopyAndDecrement();
        __host__ __device__ NoexceptCopyAndDecrement(const NoexceptCopyAndDecrement&) noexcept;

        __host__ __device__ int& operator*() const noexcept
        {
          static int x;
          return x;
        }

        __host__ __device__ NoexceptCopyAndDecrement& operator++();
        __host__ __device__ NoexceptCopyAndDecrement operator++(int);
        __host__ __device__ NoexceptCopyAndDecrement& operator--() noexcept;
        __host__ __device__ NoexceptCopyAndDecrement operator--(int) noexcept;

#  if TEST_STD_VER > 2017
        bool operator==(const NoexceptCopyAndDecrement&) const = default;
#  else
        __host__ __device__ bool operator==(const NoexceptCopyAndDecrement&) const;
        __host__ __device__ bool operator!=(const NoexceptCopyAndDecrement&) const;
#  endif
      };
      static_assert(cuda::std::bidirectional_iterator<NoexceptCopyAndDecrement>);

      static_assert(cuda::std::is_nothrow_copy_constructible_v<NoexceptCopyAndDecrement>);
      ASSERT_NOEXCEPT(cuda::std::ranges::iter_move(--cuda::std::declval<NoexceptCopyAndDecrement&>()));
      using RI = cuda::std::reverse_iterator<NoexceptCopyAndDecrement>;
      ASSERT_NOEXCEPT(iter_move(cuda::std::declval<RI>()));
    }
  }
#endif

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
