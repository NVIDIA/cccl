//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// friend iter_rvalue_reference_t<I> iter_move(const common_iterator& i)
//   noexcept(noexcept(ranges::iter_move(declval<const I&>())))
//     requires input_iterator<I>;

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"

struct IterMovingIt
{
  using value_type        = int;
  using difference_type   = int;
  explicit IterMovingIt() = default;
  __host__ __device__ IterMovingIt(const IterMovingIt&); // copyable, but this test shouldn't make copies
  IterMovingIt(IterMovingIt&&) = default;
  __host__ __device__ IterMovingIt& operator=(const IterMovingIt&);
  __host__ __device__ int& operator*() const;
  __host__ __device__ constexpr IterMovingIt& operator++()
  {
    return *this;
  }
  __host__ __device__ IterMovingIt operator++(int);
  __host__ __device__ friend constexpr int iter_move(const IterMovingIt&)
  {
    return 42;
  }

  __host__ __device__ friend bool operator==(const IterMovingIt&, cuda::std::default_sentinel_t);
#if TEST_STD_VER <= 2017
  __host__ __device__ friend bool operator==(cuda::std::default_sentinel_t, const IterMovingIt&);
  __host__ __device__ friend bool operator!=(const IterMovingIt&, cuda::std::default_sentinel_t);
  __host__ __device__ friend bool operator!=(cuda::std::default_sentinel_t, const IterMovingIt&);
#endif // TEST_STD_VER <= 2017
};
static_assert(cuda::std::input_iterator<IterMovingIt>);

__host__ __device__ constexpr bool test()
{
  {
    using It       = int*;
    using CommonIt = cuda::std::common_iterator<It, sentinel_wrapper<It>>;
    int a[]        = {1, 2, 3};
    CommonIt it    = CommonIt(It(a));
    static_assert(noexcept(iter_move(it)));
    static_assert(noexcept(cuda::std::ranges::iter_move(it)));
    static_assert(cuda::std::is_same_v<decltype(iter_move(it)), int&&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::iter_move(it)), int&&>);
    assert(iter_move(it) == 1);
    ++it;
    assert(iter_move(it) == 2);
  }
  {
    using It       = const int*;
    using CommonIt = cuda::std::common_iterator<It, sentinel_wrapper<It>>;
    int a[]        = {1, 2, 3};
    CommonIt it    = CommonIt(It(a));
    static_assert(noexcept(iter_move(it)));
    static_assert(noexcept(cuda::std::ranges::iter_move(it)));
    static_assert(cuda::std::is_same_v<decltype(iter_move(it)), const int&&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::iter_move(it)), const int&&>);
    assert(iter_move(it) == 1);
    ++it;
    assert(iter_move(it) == 2);
  }
#if TEST_COMPILER(MSVC)
  if (!cuda::std::is_constant_evaluated())
#endif // TEST_COMPILER(MSVC)
  {
    using It       = IterMovingIt;
    using CommonIt = cuda::std::common_iterator<It, cuda::std::default_sentinel_t>;
    CommonIt it    = CommonIt(It());
// old GCC seems to fall over the noexcept clauses here
#if !TEST_COMPILER(GCC, <, 9) // broken noexcept
    static_assert(!noexcept(iter_move(it)));
    static_assert(!noexcept(cuda::std::ranges::iter_move(it)));
#endif // no broken noexcept
    static_assert(cuda::std::is_same_v<decltype(iter_move(it)), int>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::iter_move(it)), int>);
    assert(iter_move(it) == 42);
    ++it;
    assert(iter_move(it) == 42);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
