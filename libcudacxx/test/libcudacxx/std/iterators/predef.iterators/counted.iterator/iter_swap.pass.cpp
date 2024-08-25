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

// template<indirectly_swappable<I> I2>
//   friend constexpr void
//     iter_swap(const counted_iterator& x, const counted_iterator<I2>& y)
//       noexcept(noexcept(ranges::iter_swap(x.current, y.current)));

#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <bool IsNoexcept>
class HasNoexceptIterSwap
{
  int* it_;

public:
  typedef cuda::std::input_iterator_tag iterator_category;
  typedef int value_type;
  typedef int element_type;
  typedef typename cuda::std::iterator_traits<int*>::difference_type difference_type;
  typedef int* pointer;
  typedef int& reference;

  __host__ __device__ constexpr int* base() const
  {
    return it_;
  }

  HasNoexceptIterSwap() = default;
  __host__ __device__ __host__ __device__ explicit constexpr HasNoexceptIterSwap(int* it)
      : it_(it)
  {}

  __host__ __device__ constexpr reference operator*() const
  {
    return *it_;
  }

  __host__ __device__ constexpr HasNoexceptIterSwap& operator++()
  {
    ++it_;
    return *this;
  }
  __host__ __device__ constexpr HasNoexceptIterSwap operator++(int)
  {
    HasNoexceptIterSwap tmp(*this);
    ++(*this);
    return tmp;
  }

  __host__ __device__ friend void iter_swap(const HasNoexceptIterSwap&, const HasNoexceptIterSwap&) noexcept(IsNoexcept)
  {}
};

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    auto iter1       = cpp17_input_iterator<int*>(buffer);
    auto commonIter1 = cuda::std::counted_iterator<decltype(iter1)>(iter1, 8);
    auto commonIter2 = cuda::std::counted_iterator<decltype(iter1)>(iter1, 8);
    for (auto i = 0; i < 4; ++i)
    {
      ++commonIter2;
    }
    assert(*commonIter2 == 5);
    cuda::std::ranges::iter_swap(commonIter1, commonIter2);
    assert(*commonIter1 == 5);
    assert(*commonIter2 == 1);
    cuda::std::ranges::iter_swap(commonIter2, commonIter1);
  }
  {
    auto iter1       = forward_iterator<int*>(buffer);
    auto commonIter1 = cuda::std::counted_iterator<decltype(iter1)>(iter1, 8);
    auto commonIter2 = cuda::std::counted_iterator<decltype(iter1)>(iter1, 8);
    for (auto i = 0; i < 4; ++i)
    {
      ++commonIter2;
    }
    assert(*commonIter2 == 5);
    cuda::std::ranges::iter_swap(commonIter1, commonIter2);
    assert(*commonIter1 == 5);
    assert(*commonIter2 == 1);
    cuda::std::ranges::iter_swap(commonIter2, commonIter1);
  }
  {
    auto iter1       = random_access_iterator<int*>(buffer);
    auto commonIter1 = cuda::std::counted_iterator<decltype(iter1)>(iter1, 8);
    auto commonIter2 = cuda::std::counted_iterator<decltype(iter1)>(iter1, 8);
    for (auto i = 0; i < 4; ++i)
    {
      ++commonIter2;
    }
    assert(*commonIter2 == 5);
    cuda::std::ranges::iter_swap(commonIter1, commonIter2);
    assert(*commonIter1 == 5);
    assert(*commonIter2 == 1);
    cuda::std::ranges::iter_swap(commonIter2, commonIter1);
  }

  // Test noexceptness.
  {
    static_assert(noexcept(cuda::std::ranges::iter_swap(
      cuda::std::declval<cuda::std::counted_iterator<HasNoexceptIterSwap<true>>&>(),
      cuda::std::declval<cuda::std::counted_iterator<HasNoexceptIterSwap<true>>&>())));
#if !defined(TEST_COMPILER_ICC) // broken noexcept
    static_assert(!noexcept(cuda::std::ranges::iter_swap(
      cuda::std::declval<cuda::std::counted_iterator<HasNoexceptIterSwap<false>>&>(),
      cuda::std::declval<cuda::std::counted_iterator<HasNoexceptIterSwap<false>>&>())));
#endif // !TEST_COMPILER_ICC
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
