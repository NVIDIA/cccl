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

#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#  include <cuda/std/compare>
#  include <cuda/std/iterator>

#  include "test_iterators.h"
#endif // _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

struct TrivialSortable
{
  int value;

  __host__ __device__ constexpr TrivialSortable()
      : value(0)
  {}
  __host__ __device__ constexpr TrivialSortable(int v)
      : value(v)
  {}
  __host__ __device__ friend constexpr bool operator<(const TrivialSortable& a, const TrivialSortable& b)
  {
    return a.value / 10 < b.value / 10;
  }
  __host__ __device__ static constexpr bool less(const TrivialSortable& a, const TrivialSortable& b)
  {
    return a.value < b.value;
  }
};

struct NonTrivialSortable
{
  int value;
  __host__ __device__ constexpr NonTrivialSortable()
      : value(0)
  {}
  __host__ __device__ constexpr NonTrivialSortable(int v)
      : value(v)
  {}
  __host__ __device__ constexpr NonTrivialSortable(const NonTrivialSortable& rhs)
      : value(rhs.value)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX14 NonTrivialSortable& operator=(const NonTrivialSortable& rhs)
  {
    value = rhs.value;
    return *this;
  }
  __host__ __device__ friend constexpr bool operator<(const NonTrivialSortable& a, const NonTrivialSortable& b)
  {
    return a.value / 10 < b.value / 10;
  }
  __host__ __device__ static constexpr bool less(const NonTrivialSortable& a, const NonTrivialSortable& b)
  {
    return a.value < b.value;
  }
};

struct TrivialSortableWithComp
{
  int value;
  __host__ __device__ constexpr TrivialSortableWithComp()
      : value(0)
  {}
  __host__ __device__ constexpr TrivialSortableWithComp(int v)
      : value(v)
  {}
  struct Comparator
  {
    __host__ __device__ constexpr bool
    operator()(const TrivialSortableWithComp& a, const TrivialSortableWithComp& b) const
    {
      return a.value / 10 < b.value / 10;
    }
  };
  static __host__ __device__ constexpr bool less(const TrivialSortableWithComp& a, const TrivialSortableWithComp& b)
  {
    return a.value < b.value;
  }
};

struct NonTrivialSortableWithComp
{
  int value;
  __host__ __device__ constexpr NonTrivialSortableWithComp()
      : value(0)
  {}
  __host__ __device__ constexpr NonTrivialSortableWithComp(int v)
      : value(v)
  {}
  __host__ __device__ constexpr NonTrivialSortableWithComp(const NonTrivialSortableWithComp& rhs)
      : value(rhs.value)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX14 NonTrivialSortableWithComp& operator=(const NonTrivialSortableWithComp& rhs)
  {
    value = rhs.value;
    return *this;
  }
  struct Comparator
  {
    __host__ __device__ constexpr bool
    operator()(const NonTrivialSortableWithComp& a, const NonTrivialSortableWithComp& b) const
    {
      return a.value / 10 < b.value / 10;
    }
  };
  static __host__ __device__ constexpr bool
  less(const NonTrivialSortableWithComp& a, const NonTrivialSortableWithComp& b)
  {
    return a.value < b.value;
  }
};

static_assert(cuda::std::is_trivially_copyable<TrivialSortable>::value, "");
static_assert(cuda::std::is_trivially_copyable<TrivialSortableWithComp>::value, "");
static_assert(!cuda::std::is_trivially_copyable<NonTrivialSortable>::value, "");
static_assert(!cuda::std::is_trivially_copyable<NonTrivialSortableWithComp>::value, "");

#if TEST_STD_VER >= 2020
struct TracedCopy
{
  int copied = 0;
  int data   = 0;

  constexpr TracedCopy() = default;
  __host__ __device__ constexpr TracedCopy(int i)
      : data(i)
  {}
  __host__ __device__ constexpr TracedCopy(const TracedCopy& other)
      : copied(other.copied + 1)
      , data(other.data)
  {}

  __host__ __device__ constexpr TracedCopy(TracedCopy&& other)            = delete;
  __host__ __device__ constexpr TracedCopy& operator=(TracedCopy&& other) = delete;

  __host__ __device__ constexpr TracedCopy& operator=(const TracedCopy& other)
  {
    copied = other.copied + 1;
    data   = other.data;
    return *this;
  }

  __host__ __device__ constexpr bool copiedOnce() const
  {
    return copied == 1;
  }

  __host__ __device__ constexpr bool operator==(const TracedCopy& o) const
  {
    return data == o.data;
  }
#  ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
  __host__ __device__ constexpr auto operator<=>(const TracedCopy& o) const
  {
    return data <=> o.data;
  }
#  else // ^^^ <=> ^^^ / vvv no <=> vvv
  __host__ __device__ constexpr auto operator<(const TracedCopy& o) const
  {
    return data < o.data;
  }
  __host__ __device__ constexpr auto operator<=(const TracedCopy& o) const
  {
    return data <= o.data;
  }
  __host__ __device__ constexpr auto operator>(const TracedCopy& o) const
  {
    return data > o.data;
  }
  __host__ __device__ constexpr auto operator>=(const TracedCopy& o) const
  {
    return data >= o.data;
  }
#  endif // !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
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

  __host__ __device__ constexpr NonBorrowedRange(int* d, cuda::std::size_t s)
      : data_{d}
      , size_{s}
  {}

  __host__ __device__ constexpr Iter begin() const
  {
    return Iter{data_};
  };
  __host__ __device__ constexpr Sent end() const
  {
    return Sent{Iter{data_ + size_}};
  };
};
#endif // TEST_STD_VER >= 2020

#endif // SORTABLE_HELPERS_H
