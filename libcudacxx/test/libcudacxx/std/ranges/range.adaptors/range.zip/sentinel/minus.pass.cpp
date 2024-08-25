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

// template <bool OtherConst>
// requires(sized_sentinel_for<sentinel_t<maybe-const<Const, Views>>,
//                             iterator_t<maybe-const<OtherConst, Views>>>&&...)
// friend constexpr common_type_t<range_difference_t<maybe-const<OtherConst, Views>>...>
//   operator-(const iterator<OtherConst>&, const sentinel&)
//
// template <bool OtherConst>
// requires(sized_sentinel_for<sentinel_t<maybe-const<Const, Views>>,
//                             iterator_t<maybe-const<OtherConst, Views>>>&&...)
// friend constexpr common_type_t<range_difference_t<maybe-const<OtherConst, Views>>...>
//   operator-(const sentinel&, const iterator<OtherConst>&)

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/functional>
#include <cuda/std/ranges>
#include <cuda/std/tuple>

#include "../types.h"

template <class Base = int*>
struct convertible_forward_sized_iterator
{
  Base it_ = nullptr;

  using iterator_category = cuda::std::forward_iterator_tag;
  using value_type        = int;
  using difference_type   = intptr_t;

  convertible_forward_sized_iterator() = default;
  __host__ __device__ constexpr convertible_forward_sized_iterator(Base it)
      : it_(it)
  {}

  template <class U, cuda::std::enable_if_t<cuda::std::convertible_to<U, Base>, int> = 0>
  __host__ __device__ constexpr convertible_forward_sized_iterator(const convertible_forward_sized_iterator<U>& it)
      : it_(it.it_)
  {}

  __host__ __device__ constexpr decltype(*Base{}) operator*() const
  {
    return *it_;
  }

  __host__ __device__ constexpr convertible_forward_sized_iterator& operator++()
  {
    ++it_;
    return *this;
  }
  __host__ __device__ constexpr convertible_forward_sized_iterator operator++(int)
  {
    return forward_sized_iterator<Base>(it_++);
  }

#if defined(TEST_COMPILER_GCC) && __GNUC__ < 9 // Old gcc has issues establishing the conversion sequence
  __host__ __device__ constexpr convertible_forward_sized_iterator(forward_sized_iterator<Base> it)
      : it_(it.it_)
  {}
#endif // defined(TEST_COMPILER_GCC) && __GNUC__ < 9

#if TEST_STD_VER >= 2020
  __host__ __device__ friend constexpr bool
  operator==(const convertible_forward_sized_iterator&, const convertible_forward_sized_iterator&) = default;
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
  __host__ __device__ friend constexpr bool
  operator==(const convertible_forward_sized_iterator& lhs, const convertible_forward_sized_iterator& rhs)
  {
    return lhs.it_ == rhs.it_;
  }
  __host__ __device__ friend constexpr bool
  operator!=(const convertible_forward_sized_iterator& lhs, const convertible_forward_sized_iterator& rhs)
  {
    return lhs.it_ != rhs.it_;
  }
#endif // TEST_STD_VER <= 2017

  __host__ __device__ friend constexpr difference_type
  operator-(const convertible_forward_sized_iterator& x, const convertible_forward_sized_iterator& y)
  {
    return x.it_ - y.it_;
  }
};
static_assert(cuda::std::forward_iterator<convertible_forward_sized_iterator<>>);

template <class Base>
struct convertible_sized_sentinel
{
  Base base_;
  explicit convertible_sized_sentinel() = default;
  __host__ __device__ constexpr convertible_sized_sentinel(const Base& it)
      : base_(it)
  {}

  template <class U, cuda::std::enable_if_t<cuda::std::convertible_to<U, Base>, int> = 0>
  __host__ __device__ constexpr convertible_sized_sentinel(const convertible_sized_sentinel<U>& other)
      : base_(other.base_)
  {}

  template <class U,
            cuda::std::enable_if_t<cuda::std::convertible_to<Base, U> || cuda::std::convertible_to<U, Base>, int> = 0>
  __host__ __device__ friend constexpr bool operator==(const convertible_sized_sentinel& s, const U& base)
  {
    return s.base_ == base;
  }
#if TEST_STD_VER <= 2017
  template <class U,
            cuda::std::enable_if_t<cuda::std::convertible_to<Base, U> || cuda::std::convertible_to<U, Base>, int> = 0>
  __host__ __device__ friend constexpr bool operator==(const U& base, const convertible_sized_sentinel& s)
  {
    return s.base_ == base;
  }
  template <class U,
            cuda::std::enable_if_t<cuda::std::convertible_to<Base, U> || cuda::std::convertible_to<U, Base>, int> = 0>
  __host__ __device__ friend constexpr bool operator!=(const convertible_sized_sentinel& s, const U& base)
  {
    return s.base_ != base;
  }
  template <class U,
            cuda::std::enable_if_t<cuda::std::convertible_to<Base, U> || cuda::std::convertible_to<U, Base>, int> = 0>
  __host__ __device__ friend constexpr bool operator!=(const U& base, const convertible_sized_sentinel& s)
  {
    return s.base_ != base;
  }
#endif // TEST_STD_VER <= 2017

  template <class U,
            cuda::std::enable_if_t<cuda::std::convertible_to<Base, U> || cuda::std::convertible_to<U, Base>, int> = 0>
  __host__ __device__ friend constexpr auto operator-(const convertible_sized_sentinel& s, const U& i)
  {
    return s.base_ - i;
  }
  template <class U,
            cuda::std::enable_if_t<cuda::std::convertible_to<Base, U> || cuda::std::convertible_to<U, Base>, int> = 0>
  __host__ __device__ friend constexpr auto operator-(const U& i, const convertible_sized_sentinel& s)
  {
    return i - s.base_;
  }
};
static_assert(cuda::std::sized_sentinel_for<convertible_sized_sentinel<convertible_forward_sized_iterator<>>,
                                            convertible_forward_sized_iterator<>>);
static_assert(cuda::std::sized_sentinel_for<convertible_sized_sentinel<convertible_forward_sized_iterator<const int*>>,
                                            convertible_forward_sized_iterator<int*>>);
static_assert(cuda::std::sized_sentinel_for<convertible_sized_sentinel<convertible_forward_sized_iterator<int*>>,
                                            convertible_forward_sized_iterator<const int*>>);

struct ConstCompatibleForwardSized : IntBufferView
{
#if defined(TEST_COMPILER_NVRTC) // nvbug 3961621
  template <cuda::std::size_t N>
  __host__ __device__ constexpr ConstCompatibleForwardSized(int (&b)[N])
      : IntBufferView(b)
  {}
#else
  using IntBufferView::IntBufferView;
#endif

  using iterator       = convertible_forward_sized_iterator<int*>;
  using const_iterator = convertible_forward_sized_iterator<const int*>;

  __host__ __device__ constexpr iterator begin()
  {
    return {buffer_};
  }
  __host__ __device__ constexpr const_iterator begin() const
  {
    return {buffer_};
  }
  __host__ __device__ constexpr convertible_sized_sentinel<iterator> end()
  {
    return iterator{buffer_ + size_};
  }
  __host__ __device__ constexpr convertible_sized_sentinel<const_iterator> end() const
  {
    return const_iterator{buffer_ + size_};
  }
};

// clang-format off
template <class T, class U>
inline constexpr bool HasMinus = cuda::std::invocable<cuda::std::minus<>,const T&, const U&>;

template <class T>
inline constexpr bool SentinelHasMinus = HasMinus<cuda::std::ranges::sentinel_t<T>, cuda::std::ranges::iterator_t<T>>;
// clang-format on

__host__ __device__ constexpr bool test()
{
  int buffer1[5] = {1, 2, 3, 4, 5};

  {
    // simple-view
    cuda::std::ranges::zip_view v{ForwardSizedNonCommon(buffer1)};
    static_assert(!cuda::std::ranges::common_range<decltype(v)>);
    static_assert(simple_view<decltype(v)>);

    auto it = v.begin();
    auto st = v.end();
    assert(st - it == 5);
    assert(st - cuda::std::ranges::next(it, 1) == 4);

    assert(it - st == -5);
    assert(cuda::std::ranges::next(it, 1) - st == -4);
    static_assert(SentinelHasMinus<decltype(v)>);
  }

  {
    // shortest range
    cuda::std::ranges::zip_view v(cuda::std::views::iota(0, 3), ForwardSizedNonCommon(buffer1));
    static_assert(!cuda::std::ranges::common_range<decltype(v)>);
    auto it = v.begin();
    auto st = v.end();
    assert(st - it == 3);
    assert(st - cuda::std::ranges::next(it, 1) == 2);

    assert(it - st == -3);
    assert(cuda::std::ranges::next(it, 1) - st == -2);
    static_assert(SentinelHasMinus<decltype(v)>);
  }

  {
    // underlying sentinel does not model sized_sentinel_for
    cuda::std::ranges::zip_view v(cuda::std::views::iota(0, 3), NonSizedRandomAccessView{buffer1});
    static_assert(!cuda::std::ranges::common_range<decltype(v)>);
    static_assert(!SentinelHasMinus<decltype(v)>);
  }

  {
    // const imcompatible:
    // underlying const sentinels cannot substract underlying iterators
    // underlying sentinels cannot substract underlying const iterators
    cuda::std::ranges::zip_view v(NonSimpleForwardSizedNonCommon{buffer1});
    static_assert(!cuda::std::ranges::common_range<decltype(v)>);
    static_assert(!simple_view<decltype(v)>);

    using Iter      = cuda::std::ranges::iterator_t<decltype(v)>;
    using ConstIter = cuda::std::ranges::iterator_t<const decltype(v)>;
    static_assert(!cuda::std::is_same_v<Iter, ConstIter>);
    using Sentinel      = cuda::std::ranges::sentinel_t<decltype(v)>;
    using ConstSentinel = cuda::std::ranges::sentinel_t<const decltype(v)>;
    static_assert(!cuda::std::is_same_v<Sentinel, ConstSentinel>);

    static_assert(HasMinus<Iter, Sentinel>);
    static_assert(HasMinus<Sentinel, Iter>);
    static_assert(HasMinus<ConstIter, ConstSentinel>);
    static_assert(HasMinus<ConstSentinel, ConstIter>);
    auto it       = v.begin();
    auto const_it = cuda::std::as_const(v).begin();
    auto st       = v.end();
    auto const_st = cuda::std::as_const(v).end();
    assert(it - st == -5);
    assert(st - it == 5);
    assert(const_it - const_st == -5);
    assert(const_st - const_it == 5);

    static_assert(!HasMinus<Iter, ConstSentinel>);
    static_assert(!HasMinus<ConstSentinel, Iter>);
    static_assert(!HasMinus<ConstIter, Sentinel>);
    static_assert(!HasMinus<Sentinel, ConstIter>);
  }

  {
    // const compatible allow non-const to const conversion
    cuda::std::ranges::zip_view v(ConstCompatibleForwardSized{buffer1});
    static_assert(!cuda::std::ranges::common_range<decltype(v)>);
    static_assert(!simple_view<decltype(v)>);

    using Iter      = cuda::std::ranges::iterator_t<decltype(v)>;
    using ConstIter = cuda::std::ranges::iterator_t<const decltype(v)>;
    static_assert(!cuda::std::is_same_v<Iter, ConstIter>);
    using Sentinel      = cuda::std::ranges::sentinel_t<decltype(v)>;
    using ConstSentinel = cuda::std::ranges::sentinel_t<const decltype(v)>;
    static_assert(!cuda::std::is_same_v<Sentinel, ConstSentinel>);

    static_assert(HasMinus<Iter, Sentinel>);
    static_assert(HasMinus<Sentinel, Iter>);
    static_assert(HasMinus<ConstIter, ConstSentinel>);
    static_assert(HasMinus<ConstSentinel, ConstIter>);
    static_assert(HasMinus<Iter, ConstSentinel>);
    static_assert(HasMinus<ConstSentinel, Iter>);
    static_assert(HasMinus<ConstIter, Sentinel>);
    static_assert(HasMinus<Sentinel, ConstIter>);

    auto it       = v.begin();
    auto const_it = cuda::std::as_const(v).begin();
    auto st       = v.end();
    auto const_st = cuda::std::as_const(v).end();

    assert(it - st == -5);
    assert(st - it == 5);
    assert(const_it - const_st == -5);
    assert(const_st - const_it == 5);
    assert(it - const_st == -5);
    assert(const_st - it == 5);
    assert(const_it - st == -5);
    assert(st - const_it == 5);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
