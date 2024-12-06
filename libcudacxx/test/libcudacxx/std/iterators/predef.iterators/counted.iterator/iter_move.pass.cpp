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

// friend constexpr iter_rvalue_reference_t<I>
//   iter_move(const counted_iterator& i)
//     noexcept(noexcept(ranges::iter_move(i.current)))
//     requires input_iterator<I>;

#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <bool IsNoexcept>
class HasNoexceptIterMove
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

  HasNoexceptIterMove() = default;
  __host__ __device__ explicit constexpr HasNoexceptIterMove(int* it)
      : it_(it)
  {}

  __host__ __device__ constexpr reference operator*() const noexcept(IsNoexcept)
  {
    return *it_;
  }

  __host__ __device__ constexpr HasNoexceptIterMove& operator++()
  {
    ++it_;
    return *this;
  }
  __host__ __device__ constexpr HasNoexceptIterMove operator++(int)
  {
    HasNoexceptIterMove tmp(*this);
    ++(*this);
    return tmp;
  }
};

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    auto iter1       = cpp17_input_iterator<int*>(buffer);
    auto commonIter1 = cuda::std::counted_iterator<decltype(iter1)>(iter1, 8);
    assert(cuda::std::ranges::iter_move(commonIter1) == 1);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::iter_move(commonIter1)), int&&);
  }
  {
    auto iter1       = forward_iterator<int*>(buffer);
    auto commonIter1 = cuda::std::counted_iterator<decltype(iter1)>(iter1, 8);
    assert(cuda::std::ranges::iter_move(commonIter1) == 1);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::iter_move(commonIter1)), int&&);
  }
  {
    auto iter1       = random_access_iterator<int*>(buffer);
    auto commonIter1 = cuda::std::counted_iterator<decltype(iter1)>(iter1, 8);
    assert(cuda::std::ranges::iter_move(commonIter1) == 1);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::iter_move(commonIter1)), int&&);
  }

  // Test noexceptness.
  {
    static_assert(noexcept(
      cuda::std::ranges::iter_move(cuda::std::declval<cuda::std::counted_iterator<HasNoexceptIterMove<true>>>())));
#if !defined(TEST_COMPILER_ICC) // broken noexcept
    static_assert(!noexcept(
      cuda::std::ranges::iter_move(cuda::std::declval<cuda::std::counted_iterator<HasNoexceptIterMove<false>>>())));
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
