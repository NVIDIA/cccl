//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// template<indirectly_swappable<I> I2, class S2>
//   friend constexpr void iter_swap(const common_iterator& x, const common_iterator<I2, S2>& y)
//     noexcept(noexcept(ranges::iter_swap(declval<const I&>(), declval<const I2&>())));

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"

template <int K>
struct IterSwappingIt
{
  using value_type      = int;
  using difference_type = int;
  __host__ __device__ constexpr explicit IterSwappingIt(int* swaps)
      : swaps_(swaps)
  {}
  __host__ __device__ IterSwappingIt(const IterSwappingIt&); // copyable, but this test shouldn't make copies
  IterSwappingIt(IterSwappingIt&&) = default;
  __host__ __device__ IterSwappingIt& operator=(const IterSwappingIt&);
  __host__ __device__ int& operator*() const;
  __host__ __device__ constexpr IterSwappingIt& operator++()
  {
    return *this;
  }
  __host__ __device__ IterSwappingIt operator++(int);

  template <int L>
  __host__ __device__ friend constexpr int iter_swap(const IterSwappingIt<K>& lhs, const IterSwappingIt<L>& rhs)
  {
    *lhs.swaps_ += 10;
    *rhs.swaps_ += 1;
    return 42; // should be accepted but ignored
  }

  __host__ __device__ friend bool operator==(const IterSwappingIt&, cuda::std::default_sentinel_t)
  {
    return true;
  }
#if TEST_STD_VER <= 2017
  __host__ __device__ friend bool operator==(cuda::std::default_sentinel_t, const IterSwappingIt&)
  {
    return true;
  };
  __host__ __device__ friend bool operator!=(const IterSwappingIt&, cuda::std::default_sentinel_t)
  {
    return false;
  };
  __host__ __device__ friend bool operator!=(cuda::std::default_sentinel_t, const IterSwappingIt&)
  {
    return false;
  };
#endif // TEST_STD_VER <= 2017

  int* swaps_ = nullptr;
};
static_assert(cuda::std::input_iterator<IterSwappingIt<0>>);
static_assert(cuda::std::indirectly_swappable<IterSwappingIt<0>, IterSwappingIt<0>>);
static_assert(cuda::std::indirectly_swappable<IterSwappingIt<0>, IterSwappingIt<1>>);

__host__ __device__ constexpr bool test()
{
  {
    using It       = int*;
    using CommonIt = cuda::std::common_iterator<It, sentinel_wrapper<It>>;
    static_assert(cuda::std::indirectly_swappable<CommonIt, CommonIt>);

    int a[]     = {1, 2, 3};
    CommonIt it = CommonIt(It(a));
    CommonIt jt = CommonIt(It(a + 1));
    ASSERT_NOEXCEPT(iter_swap(it, jt));
    ASSERT_SAME_TYPE(decltype(iter_swap(it, jt)), void);
    iter_swap(it, jt);
    assert(a[0] == 2);
    assert(a[1] == 1);
  }
  {
    using It       = const int*;
    using CommonIt = cuda::std::common_iterator<It, sentinel_wrapper<It>>;
    static_assert(!cuda::std::indirectly_swappable<CommonIt, CommonIt>);
  }
  {
    using It       = IterSwappingIt<0>;
    using CommonIt = cuda::std::common_iterator<It, cuda::std::default_sentinel_t>;
    static_assert(cuda::std::indirectly_swappable<CommonIt, CommonIt>);

    int iswaps  = 100;
    int jswaps  = 100;
    CommonIt it = CommonIt(It(&iswaps));
    CommonIt jt = CommonIt(It(&jswaps));
#if !defined(TEST_COMPILER_ICC) // broken noexcept
    ASSERT_NOT_NOEXCEPT(iter_swap(it, jt));
#endif // !TEST_COMPILER_ICC
    ASSERT_SAME_TYPE(decltype(iter_swap(it, jt)), void);
    iter_swap(it, jt); // lvalue iterators
    assert(iswaps == 110);
    assert(jswaps == 101);
    iter_swap(CommonIt(It(&iswaps)), CommonIt(It(&jswaps))); // rvalue iterators
    assert(iswaps == 120);
    assert(jswaps == 102);
    cuda::std::ranges::iter_swap(it, jt);
    assert(iswaps == 130);
    assert(jswaps == 103);
  }
  {
    using It = IterSwappingIt<0>;
    using Jt = IterSwappingIt<1>;
    static_assert(cuda::std::indirectly_swappable<It, Jt>);
    using CommonIt = cuda::std::common_iterator<It, cuda::std::default_sentinel_t>;
    using CommonJt = cuda::std::common_iterator<Jt, cuda::std::default_sentinel_t>;
    static_assert(cuda::std::indirectly_swappable<CommonIt, CommonJt>);

    int iswaps  = 100;
    int jswaps  = 100;
    CommonIt it = CommonIt(It(&iswaps));
    CommonJt jt = CommonJt(Jt(&jswaps));
#if !defined(TEST_COMPILER_ICC) // broken noexcept
    ASSERT_NOT_NOEXCEPT(iter_swap(it, jt));
#endif // !TEST_COMPILER_ICC
    ASSERT_SAME_TYPE(decltype(iter_swap(it, jt)), void);
    iter_swap(it, jt); // lvalue iterators
    assert(iswaps == 110);
    assert(jswaps == 101);
    iter_swap(CommonIt(It(&iswaps)), CommonJt(Jt(&jswaps))); // rvalue iterators
    assert(iswaps == 120);
    assert(jswaps == 102);
    cuda::std::ranges::iter_swap(it, jt);
    assert(iswaps == 130);
    assert(jswaps == 103);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
