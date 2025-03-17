//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <iterator>
//
// constexpr iterator_type base() const; // Until C++20
// constexpr const Iterator& base() const & noexcept; // From C++20
// constexpr Iterator base() &&; // From C++20

#include <cuda/std/iterator>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "test_macros.h"

struct MoveOnlyIterator
{
  using It = int*;

  It it_;

  using iterator_category = cuda::std::input_iterator_tag;
  using value_type        = int;
  using difference_type   = cuda::std::ptrdiff_t;
  using reference         = int&;

  __host__ __device__ constexpr explicit MoveOnlyIterator(It it)
      : it_(it)
  {}
  MoveOnlyIterator(MoveOnlyIterator&&)                 = default;
  MoveOnlyIterator& operator=(MoveOnlyIterator&&)      = default;
  MoveOnlyIterator(const MoveOnlyIterator&)            = delete;
  MoveOnlyIterator& operator=(const MoveOnlyIterator&) = delete;

  __host__ __device__ constexpr reference operator*() const
  {
    return *it_;
  }

  __host__ __device__ constexpr MoveOnlyIterator& operator++()
  {
    ++it_;
    return *this;
  }
  __host__ __device__ constexpr MoveOnlyIterator operator++(int)
  {
    return MoveOnlyIterator(it_++);
  }

  __host__ __device__ friend constexpr bool operator==(const MoveOnlyIterator& x, const MoveOnlyIterator& y)
  {
    return x.it_ == y.it_;
  }
  __host__ __device__ friend constexpr bool operator!=(const MoveOnlyIterator& x, const MoveOnlyIterator& y)
  {
    return x.it_ != y.it_;
  }

  __host__ __device__ friend constexpr It base(const MoveOnlyIterator& i)
  {
    return i.it_;
  }
};

static_assert(cuda::std::input_iterator<MoveOnlyIterator>, "");
static_assert(!cuda::std::is_copy_constructible<MoveOnlyIterator>::value, "");

template <class It>
__host__ __device__ constexpr void test_one()
{
  // Non-const lvalue.
  {
    int a[] = {1, 2, 3};

    auto i = cuda::std::move_iterator<It>(It(a));
    static_assert(cuda::std::is_same_v<decltype(i.base()), const It&>);
    static_assert(noexcept(i.base()));

    assert(i.base() == It(a));

    ++i;
    assert(i.base() == It(a + 1));
  }

  // Const lvalue.
  {
    int a[] = {1, 2, 3};

    const auto i = cuda::std::move_iterator<It>(It(a));
    static_assert(cuda::std::is_same_v<decltype(i.base()), const It&>);
    static_assert(noexcept(i.base()));
    assert(i.base() == It(a));
  }

  // Rvalue.
  {
    int a[] = {1, 2, 3};

    auto i = cuda::std::move_iterator<It>(It(a));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::move(i).base()), It>);
    assert(cuda::std::move(i).base() == It(a));
  }
}

__host__ __device__ constexpr bool test()
{
  test_one<cpp17_input_iterator<int*>>();
  test_one<forward_iterator<int*>>();
  test_one<bidirectional_iterator<int*>>();
  test_one<random_access_iterator<int*>>();
  test_one<int*>();
  test_one<const int*>();
  test_one<contiguous_iterator<int*>>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
