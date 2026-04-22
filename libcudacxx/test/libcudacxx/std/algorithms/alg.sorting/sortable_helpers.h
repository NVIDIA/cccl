//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SORTABLE_HELPERS_H
#define SORTABLE_HELPERS_H

#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/compare>
#  include <cuda/std/iterator>

#  include "test_iterators.h"
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

struct TrivialSortable
{
  int value;

  TEST_FUNC constexpr TrivialSortable()
      : value(0)
  {}
  TEST_FUNC constexpr TrivialSortable(int v)
      : value(v)
  {}
  TEST_FUNC friend constexpr bool operator<(const TrivialSortable& a, const TrivialSortable& b)
  {
    return a.value / 10 < b.value / 10;
  }
  TEST_FUNC static constexpr bool less(const TrivialSortable& a, const TrivialSortable& b)
  {
    return a.value < b.value;
  }
};

struct NonTrivialSortable
{
  int value;
  TEST_FUNC constexpr NonTrivialSortable()
      : value(0)
  {}
  TEST_FUNC constexpr NonTrivialSortable(int v)
      : value(v)
  {}
  TEST_FUNC constexpr NonTrivialSortable(const NonTrivialSortable& rhs)
      : value(rhs.value)
  {}
  TEST_FUNC constexpr NonTrivialSortable& operator=(const NonTrivialSortable& rhs)
  {
    value = rhs.value;
    return *this;
  }
  TEST_FUNC friend constexpr bool operator<(const NonTrivialSortable& a, const NonTrivialSortable& b)
  {
    return a.value / 10 < b.value / 10;
  }
  TEST_FUNC static constexpr bool less(const NonTrivialSortable& a, const NonTrivialSortable& b)
  {
    return a.value < b.value;
  }
};

struct TrivialSortableWithComp
{
  int value;
  TEST_FUNC constexpr TrivialSortableWithComp()
      : value(0)
  {}
  TEST_FUNC constexpr TrivialSortableWithComp(int v)
      : value(v)
  {}
  struct Comparator
  {
    TEST_FUNC constexpr bool operator()(const TrivialSortableWithComp& a, const TrivialSortableWithComp& b) const
    {
      return a.value / 10 < b.value / 10;
    }
  };
  static TEST_FUNC constexpr bool less(const TrivialSortableWithComp& a, const TrivialSortableWithComp& b)
  {
    return a.value < b.value;
  }
};

struct NonTrivialSortableWithComp
{
  int value;
  TEST_FUNC constexpr NonTrivialSortableWithComp()
      : value(0)
  {}
  TEST_FUNC constexpr NonTrivialSortableWithComp(int v)
      : value(v)
  {}
  TEST_FUNC constexpr NonTrivialSortableWithComp(const NonTrivialSortableWithComp& rhs)
      : value(rhs.value)
  {}
  TEST_FUNC constexpr NonTrivialSortableWithComp& operator=(const NonTrivialSortableWithComp& rhs)
  {
    value = rhs.value;
    return *this;
  }
  struct Comparator
  {
    TEST_FUNC constexpr bool operator()(const NonTrivialSortableWithComp& a, const NonTrivialSortableWithComp& b) const
    {
      return a.value / 10 < b.value / 10;
    }
  };
  static TEST_FUNC constexpr bool less(const NonTrivialSortableWithComp& a, const NonTrivialSortableWithComp& b)
  {
    return a.value < b.value;
  }
};

static_assert(cuda::std::is_trivially_copyable<TrivialSortable>::value);
static_assert(cuda::std::is_trivially_copyable<TrivialSortableWithComp>::value);
static_assert(!cuda::std::is_trivially_copyable<NonTrivialSortable>::value);
static_assert(!cuda::std::is_trivially_copyable<NonTrivialSortableWithComp>::value);

#if TEST_STD_VER >= 2020
struct TracedCopy
{
  int copied = 0;
  int data   = 0;

  constexpr TracedCopy() = default;
  TEST_FUNC constexpr TracedCopy(int i)
      : data(i)
  {}
  TEST_FUNC constexpr TracedCopy(const TracedCopy& other)
      : copied(other.copied + 1)
      , data(other.data)
  {}

  TEST_FUNC constexpr TracedCopy(TracedCopy&& other)            = delete;
  TEST_FUNC constexpr TracedCopy& operator=(TracedCopy&& other) = delete;

  TEST_FUNC constexpr TracedCopy& operator=(const TracedCopy& other)
  {
    copied = other.copied + 1;
    data   = other.data;
    return *this;
  }

  TEST_FUNC constexpr bool copiedOnce() const
  {
    return copied == 1;
  }

  TEST_FUNC constexpr bool operator==(const TracedCopy& o) const
  {
    return data == o.data;
  }
#  if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  TEST_FUNC constexpr auto operator<=>(const TracedCopy& o) const
  {
    return data <=> o.data;
  }
#  else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv
  TEST_FUNC constexpr auto operator<(const TracedCopy& o) const
  {
    return data < o.data;
  }
  TEST_FUNC constexpr auto operator<=(const TracedCopy& o) const
  {
    return data <= o.data;
  }
  TEST_FUNC constexpr auto operator>(const TracedCopy& o) const
  {
    return data > o.data;
  }
  TEST_FUNC constexpr auto operator>=(const TracedCopy& o) const
  {
    return data >= o.data;
  }
#  endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
};

template <class Iter>
struct NonBorrowedRange
{
  int* data_;
  cuda::std::size_t size_;

  // TODO: some algorithms calls std::__copy
  // std::__copy(contiguous_iterator<int*>, sentinel_wrapper<contiguous_iterator<int*>>, contiguous_iterator<int*>)
  // doesn't seem to work. It seems that it unwraps contiguous_iterator<int*> into int*, and then it failed because
  // there is no == between int* and sentinel_wrapper<contiguous_iterator<int*>>
  using Sent = cuda::std::conditional_t<cuda::std::contiguous_iterator<Iter>, Iter, sentinel_wrapper<Iter>>;

  TEST_FUNC constexpr NonBorrowedRange(int* d, cuda::std::size_t s)
      : data_{d}
      , size_{s}
  {}

  TEST_FUNC constexpr Iter begin() const
  {
    return Iter{data_};
  };
  TEST_FUNC constexpr Sent end() const
  {
    return Sent{Iter{data_ + size_}};
  };
};
#endif // TEST_STD_VER >= 2020

#endif // SORTABLE_HELPERS_H
