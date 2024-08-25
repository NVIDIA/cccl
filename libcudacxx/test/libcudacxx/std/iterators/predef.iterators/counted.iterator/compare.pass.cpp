//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// template<common_with<I> I2>
//   friend constexpr bool operator==(
//     const counted_iterator& x, const counted_iterator<I2>& y);
// friend constexpr bool operator==(
//   const counted_iterator& x, default_sentinel_t);

#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

// This iterator is common_with forward_iterator but NOT comparable with it.
template <class It>
class CommonWithForwardIter
{
  It it_;

public:
  typedef cuda::std::input_iterator_tag iterator_category;
  typedef typename cuda::std::iterator_traits<It>::value_type value_type;
  typedef typename cuda::std::iterator_traits<It>::difference_type difference_type;
  typedef It pointer;
  typedef typename cuda::std::iterator_traits<It>::reference reference;

  __host__ __device__ constexpr It base() const
  {
    return it_;
  }

  CommonWithForwardIter() = default;
  __host__ __device__ explicit constexpr CommonWithForwardIter(It it)
      : it_(it)
  {}
  __host__ __device__ constexpr CommonWithForwardIter(const forward_iterator<It>& it)
      : it_(it.base())
  {}

  __host__ __device__ constexpr reference operator*() const
  {
    return *it_;
  }

  __host__ __device__ constexpr CommonWithForwardIter& operator++()
  {
    ++it_;
    return *this;
  }
  __host__ __device__ constexpr CommonWithForwardIter operator++(int)
  {
    CommonWithForwardIter tmp(*this);
    ++(*this);
    return tmp;
  }
};

struct InputOrOutputArchetype
{
  using difference_type = int;

  int* ptr;

  __host__ __device__ constexpr int operator*()
  {
    return *ptr;
  }
  __host__ __device__ constexpr void operator++(int)
  {
    ++ptr;
  }
  __host__ __device__ constexpr InputOrOutputArchetype& operator++()
  {
    ++ptr;
    return *this;
  }
};

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    {
      cuda::std::counted_iterator iter1(forward_iterator<int*>(buffer), 8);
      cuda::std::counted_iterator iter2(CommonWithForwardIter<int*>(buffer), 8);

      assert(iter1 == iter2);
      assert(iter2 == iter1);
      ++iter1;
      assert(iter1 != iter2);
      assert(iter2 != iter1);
    }
  }

  {
    {
      cuda::std::counted_iterator iter(cpp20_input_iterator<int*>(buffer), 8);

      assert(iter != cuda::std::default_sentinel);
      assert(cuda::std::default_sentinel == cuda::std::ranges::next(cuda::std::move(iter), 8));
    }
    {
      cuda::std::counted_iterator iter(forward_iterator<int*>(buffer), 8);

      assert(iter != cuda::std::default_sentinel);
      assert(cuda::std::default_sentinel == cuda::std::ranges::next(iter, 8));
    }
    {
      cuda::std::counted_iterator iter(random_access_iterator<int*>(buffer), 8);

      assert(iter != cuda::std::default_sentinel);
      assert(cuda::std::default_sentinel == cuda::std::ranges::next(iter, 8));
    }
    {
      cuda::std::counted_iterator iter(InputOrOutputArchetype{buffer}, 8);

      assert(iter != cuda::std::default_sentinel);
      assert(cuda::std::default_sentinel == cuda::std::ranges::next(iter, 8));
    }
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
