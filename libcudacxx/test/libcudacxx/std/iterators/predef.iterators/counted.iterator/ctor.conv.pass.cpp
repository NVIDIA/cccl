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

// template<class I2>
//   requires convertible_to<const I2&, I>
//     constexpr counted_iterator(const counted_iterator<I2>& x);

#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class T>
class ConvertibleTo
{
  int* it_;

public:
  typedef cuda::std::input_iterator_tag iterator_category;
  typedef int value_type;
  typedef typename cuda::std::iterator_traits<int*>::difference_type difference_type;
  typedef int* pointer;
  typedef int& reference;

  __host__ __device__ constexpr int* base() const
  {
    return it_;
  }

  ConvertibleTo() = default;
  __host__ __device__ explicit constexpr ConvertibleTo(int* it)
      : it_(it)
  {}

  __host__ __device__ constexpr reference operator*() const
  {
    return *it_;
  }

  __host__ __device__ constexpr ConvertibleTo& operator++()
  {
    ++it_;
    return *this;
  }
  __host__ __device__ constexpr ConvertibleTo operator++(int)
  {
    ConvertibleTo tmp(*this);
    ++(*this);
    return tmp;
  }

  __host__ __device__ constexpr operator T() const
  {
    return T(it_);
  }
};

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    static_assert(cuda::std::is_constructible_v<cuda::std::counted_iterator<forward_iterator<int*>>,
                                                cuda::std::counted_iterator<forward_iterator<int*>>>);
    static_assert(!cuda::std::is_constructible_v<cuda::std::counted_iterator<forward_iterator<int*>>,
                                                 cuda::std::counted_iterator<random_access_iterator<int*>>>);
  }
  {
    cuda::std::counted_iterator iter1(ConvertibleTo<forward_iterator<int*>>{buffer}, 8);
    cuda::std::counted_iterator<forward_iterator<int*>> iter2(iter1);
    assert(iter2.base() == forward_iterator<int*>{buffer});
    assert(iter2.count() == 8);
  }
  {
    const cuda::std::counted_iterator iter1(ConvertibleTo<forward_iterator<int*>>{buffer}, 8);
    const cuda::std::counted_iterator<forward_iterator<int*>> iter2(iter1);
    assert(iter2.base() == forward_iterator<int*>{buffer});
    assert(iter2.count() == 8);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
