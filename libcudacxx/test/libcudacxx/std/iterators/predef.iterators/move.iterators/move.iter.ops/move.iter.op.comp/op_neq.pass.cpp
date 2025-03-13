//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// template <InputIterator Iter1, InputIterator Iter2>
//   requires HasEqualTo<Iter1, Iter2>
//   bool
//   operator!=(const move_iterator<Iter1>& x, const move_iterator<Iter2>& y);
//
//  constexpr in C++17

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

// In C++17, move_iterator's operator!= calls the underlying iterator's operator!=
// In C++20, move_iterator's operator== calls the underlying iterator's operator==
struct CustomIt
{
  using value_type        = int;
  using difference_type   = int;
  using reference         = int&;
  using pointer           = int*;
  using iterator_category = cuda::std::input_iterator_tag;
  CustomIt()              = default;
  __host__ __device__ constexpr explicit CustomIt(int* p)
      : p_(p)
  {}
  __host__ __device__ int& operator*() const;
  __host__ __device__ CustomIt& operator++();
  __host__ __device__ CustomIt operator++(int);
#if TEST_STD_VER > 2017
  __host__ __device__ friend constexpr bool operator==(const CustomIt& a, const CustomIt& b)
  {
    return a.p_ == b.p_;
  }
  __host__ __device__ friend bool operator!=(const CustomIt& a, const CustomIt& b) = delete;
#else
  __host__ __device__ friend constexpr bool operator!=(const CustomIt& a, const CustomIt& b)
  {
    return a.p_ != b.p_;
  }
#endif
  int* p_ = nullptr;
};

template <class It>
__host__ __device__ constexpr void test_one()
{
  int a[]                               = {3, 1, 4};
  const cuda::std::move_iterator<It> r1 = cuda::std::move_iterator<It>(It(a));
  const cuda::std::move_iterator<It> r2 = cuda::std::move_iterator<It>(It(a));
  const cuda::std::move_iterator<It> r3 = cuda::std::move_iterator<It>(It(a + 2));
  ASSERT_SAME_TYPE(decltype(r1 != r2), bool);
  assert(!(r1 != r1));
  assert(!(r1 != r2));
  assert(!(r2 != r1));
  assert((r1 != r3));
  assert((r3 != r1));
}

__host__ __device__ constexpr bool test()
{
  test_one<CustomIt>();
  test_one<cpp17_input_iterator<int*>>();
  test_one<forward_iterator<int*>>();
  test_one<bidirectional_iterator<int*>>();
  test_one<random_access_iterator<int*>>();
  test_one<int*>();
  test_one<const int*>();

  test_one<contiguous_iterator<int*>>();
#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  test_one<three_way_contiguous_iterator<int*>>();
#endif

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
