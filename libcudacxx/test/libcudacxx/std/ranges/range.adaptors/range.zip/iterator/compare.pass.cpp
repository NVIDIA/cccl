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

// friend constexpr bool operator==(const iterator& x, const iterator& y)
//   requires (equality_comparable<iterator_t<maybe-const<Const, Views>>> && ...);
// friend constexpr bool operator<(const iterator& x, const iterator& y)
//   requires all-random-access<Const, Views...>;
// friend constexpr bool operator>(const iterator& x, const iterator& y)
//   requires all-random-access<Const, Views...>;
// friend constexpr bool operator<=(const iterator& x, const iterator& y)
//   requires all-random-access<Const, Views...>;
// friend constexpr bool operator>=(const iterator& x, const iterator& y)
//   requires all-random-access<Const, Views...>;
// friend constexpr auto operator<=>(const iterator& x, const iterator& y)
//   requires all-random-access<Const, Views...> &&
//            (three_way_comparable<iterator_t<maybe-const<Const, Views>>> && ...);

#include <cuda/std/ranges>
#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#  include <cuda/std/compare>
#endif // !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

#include "../types.h"
#include "test_iterators.h"

// This is for testing that zip iterator never calls underlying iterator's >, >=, <=, !=.
// The spec indicates that zip iterator's >= is negating zip iterator's < instead of calling underlying iterator's >=.
// Declare all the operations >, >=, <= etc to make it satisfy random_access_iterator concept,
// but not define them. If the zip iterator's >,>=, <=, etc isn't implemented in the way defined by the standard
// but instead calling underlying iterator's >,>=,<=, we will get a linker error for the runtime tests and
// non-constant expression for the compile time tests.
struct LessThanIterator
{
  int* it_           = nullptr;
  LessThanIterator() = default;
  __host__ __device__ constexpr LessThanIterator(int* it)
      : it_(it)
  {}

  using iterator_category = cuda::std::random_access_iterator_tag;
  using value_type        = int;
  using difference_type   = intptr_t;

  __host__ __device__ constexpr int& operator*() const
  {
    return *it_;
  }
  __host__ __device__ constexpr int& operator[](difference_type n) const
  {
    return it_[n];
  }
  __host__ __device__ constexpr LessThanIterator& operator++()
  {
    ++it_;
    return *this;
  }
  __host__ __device__ constexpr LessThanIterator& operator--()
  {
    --it_;
    return *this;
  }
  __host__ __device__ constexpr LessThanIterator operator++(int)
  {
    return LessThanIterator(it_++);
  }
  __host__ __device__ constexpr LessThanIterator operator--(int)
  {
    return LessThanIterator(it_--);
  }

  __host__ __device__ constexpr LessThanIterator& operator+=(difference_type n)
  {
    it_ += n;
    return *this;
  }
  __host__ __device__ constexpr LessThanIterator& operator-=(difference_type n)
  {
    it_ -= n;
    return *this;
  }

  __host__ __device__ constexpr friend LessThanIterator operator+(LessThanIterator x, difference_type n)
  {
    x += n;
    return x;
  }
  __host__ __device__ constexpr friend LessThanIterator operator+(difference_type n, LessThanIterator x)
  {
    x += n;
    return x;
  }
  __host__ __device__ constexpr friend LessThanIterator operator-(LessThanIterator x, difference_type n)
  {
    x -= n;
    return x;
  }
  __host__ __device__ constexpr friend difference_type operator-(LessThanIterator x, LessThanIterator y)
  {
    return x.it_ - y.it_;
  }

  __host__ __device__ constexpr friend bool operator==(LessThanIterator const& x, LessThanIterator const& y)
  {
    return x.it_ == y.it_;
  }
  __host__ __device__ friend bool operator!=(LessThanIterator const& x, LessThanIterator const& y);

  __host__ __device__ constexpr friend bool operator<(LessThanIterator const& x, LessThanIterator const& y)
  {
    return x.it_ < y.it_;
  }
  __host__ __device__ friend bool operator<=(LessThanIterator const&, LessThanIterator const&);
  __host__ __device__ friend bool operator>(LessThanIterator const&, LessThanIterator const&);
  __host__ __device__ friend bool operator>=(LessThanIterator const&, LessThanIterator const&);
};
static_assert(cuda::std::random_access_iterator<LessThanIterator>);

struct SmallerThanRange : IntBufferView
{
#if defined(TEST_COMPILER_NVRTC) // nvbug 3961621
  template <cuda::std::size_t N>
  __host__ __device__ constexpr SmallerThanRange(int (&b)[N])
      : IntBufferView(b)
  {}
#else
  using IntBufferView::IntBufferView;
#endif
  __host__ __device__ constexpr LessThanIterator begin() const
  {
    return {buffer_};
  }
  __host__ __device__ constexpr LessThanIterator end() const
  {
    return {buffer_ + size_};
  }
};
static_assert(cuda::std::ranges::random_access_range<SmallerThanRange>);

struct ForwardCommonView : IntBufferView
{
#if defined(TEST_COMPILER_NVRTC) // nvbug 3961621
  template <cuda::std::size_t N>
  __host__ __device__ constexpr ForwardCommonView(int (&b)[N])
      : IntBufferView(b)
  {}
#else
  using IntBufferView::IntBufferView;
#endif
  using iterator = forward_iterator<int*>;

  __host__ __device__ constexpr iterator begin() const
  {
    return iterator(buffer_);
  }
  __host__ __device__ constexpr iterator end() const
  {
    return iterator(buffer_ + size_);
  }
};

template <class Iter1, class Iter2>
__host__ __device__ constexpr void compareOperatorTest(Iter1&& iter1, Iter2&& iter2)
{
  assert(!(iter1 < iter1));
  assert(iter1 < iter2);
  assert(!(iter2 < iter1));
  assert(iter1 <= iter1);
  assert(iter1 <= iter2);
  assert(!(iter2 <= iter1));
  assert(!(iter1 > iter1));
  assert(!(iter1 > iter2));
  assert(iter2 > iter1);
  assert(iter1 >= iter1);
  assert(!(iter1 >= iter2));
  assert(iter2 >= iter1);
  assert(iter1 == iter1);
  assert(!(iter1 == iter2));
  assert(iter2 == iter2);
  assert(!(iter1 != iter1));
  assert(iter1 != iter2);
  assert(!(iter2 != iter2));
}

template <class Iter1, class Iter2>
__host__ __device__ constexpr void inequalityOperatorsDoNotExistTest(Iter1&& iter1, Iter2&& iter2)
{
  static_assert(!cuda::std::is_invocable_v<cuda::std::less<>, Iter1, Iter2>);
  static_assert(!cuda::std::is_invocable_v<cuda::std::less_equal<>, Iter1, Iter2>);
  static_assert(!cuda::std::is_invocable_v<cuda::std::greater<>, Iter1, Iter2>);
  static_assert(!cuda::std::is_invocable_v<cuda::std::greater_equal<>, Iter1, Iter2>);
}

__host__ __device__ constexpr bool test()
{
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  {
    // Test a new-school iterator with operator<=>; the iterator should also have operator<=>.
    using It       = three_way_contiguous_iterator<int*>;
    using SubRange = cuda::std::ranges::subrange<It>;
    static_assert(cuda::std::three_way_comparable<It>);
    using R = cuda::std::ranges::zip_view<SubRange, SubRange>;
    static_assert(cuda::std::three_way_comparable<cuda::std::ranges::iterator_t<R>>);

    int a[]    = {1, 2, 3, 4};
    int b[]    = {5, 6, 7, 8, 9};
    auto r     = cuda::std::views::zip(SubRange(It(a), It(a + 4)), SubRange(It(b), It(b + 5)));
    auto iter1 = r.begin();
    auto iter2 = iter1 + 1;

    compareOperatorTest(iter1, iter2);

    assert((iter1 <=> iter2) == cuda::std::strong_ordering::less);
    assert((iter1 <=> iter1) == cuda::std::strong_ordering::equal);
    assert((iter2 <=> iter1) == cuda::std::strong_ordering::greater);
  }
#endif // !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

  {
    // Test an old-school iterator with no operator<=>; the transform iterator shouldn't have
    // operator<=> either.
    using It       = random_access_iterator<int*>;
    using Subrange = cuda::std::ranges::subrange<It>;
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
    static_assert(!cuda::std::three_way_comparable<It>);
    using R = cuda::std::ranges::zip_view<Subrange, Subrange>;
    static_assert(!cuda::std::three_way_comparable<cuda::std::ranges::iterator_t<R>>);
#endif // !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

    int a[] = {1, 2, 3, 4};
    int b[] = {5, 6, 7, 8, 9};
    Subrange sub_a(It(a), It(a + 4));
    Subrange sub_b(It(b), It(b + 5));
    auto r     = cuda::std::views::zip(sub_a, sub_b);
    auto iter1 = r.begin();
    auto iter2 = iter1 + 1;

    compareOperatorTest(iter1, iter2);
  }

  {
    // non random_access_range
    int buffer1[1] = {1};
    int buffer2[2] = {1, 2};

    cuda::std::ranges::zip_view v{InputCommonView(buffer1), InputCommonView(buffer2)};
    using View = decltype(v);
    static_assert(!cuda::std::ranges::forward_range<View>);
    static_assert(cuda::std::ranges::input_range<View>);
    static_assert(cuda::std::ranges::common_range<View>);

    auto it1 = v.begin();
    auto it2 = v.end();
    assert(it1 != it2);

    ++it1;
    assert(it1 == it2);

    inequalityOperatorsDoNotExistTest(it1, it2);
  }

  {
    // in this case sentinel is computed by getting each of the underlying sentinel, so only one
    // underlying iterator is comparing equal
    int buffer1[1] = {1};
    int buffer2[2] = {1, 2};
    cuda::std::ranges::zip_view v{ForwardCommonView(buffer1), ForwardCommonView(buffer2)};
    using View = decltype(v);
    static_assert(cuda::std::ranges::common_range<View>);
    static_assert(!cuda::std::ranges::bidirectional_range<View>);

    auto it1 = v.begin();
    auto it2 = v.end();
    assert(it1 != it2);

    ++it1;
    // it1:  <buffer1 + 1, buffer2 + 1>
    // it2:  <buffer1 + 1, buffer2 + 2>
    assert(it1 == it2);

    inequalityOperatorsDoNotExistTest(it1, it2);
  }

  {
    // only < and == are needed
    int a[]    = {1, 2, 3, 4};
    int b[]    = {5, 6, 7, 8, 9};
    auto r     = cuda::std::views::zip(SmallerThanRange(a), SmallerThanRange(b));
    auto iter1 = r.begin();
    auto iter2 = iter1 + 1;

    compareOperatorTest(iter1, iter2);
  }

  {
    // underlying iterator does not support ==
    using IterNoEqualView = BasicView<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>;
    int buffer[]          = {1};
    cuda::std::ranges::zip_view r(IterNoEqualView{buffer});
    auto it    = r.begin();
    using Iter = decltype(it);
    static_assert(!cuda::std::invocable<cuda::std::equal_to<>, Iter, Iter>);
    inequalityOperatorsDoNotExistTest(it, it);
  }
  return true;
}

int main(int, char**)
{
  test();
#if defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // defined(_LIBCUDACXX_ADDRESSOF)

  return 0;
}
