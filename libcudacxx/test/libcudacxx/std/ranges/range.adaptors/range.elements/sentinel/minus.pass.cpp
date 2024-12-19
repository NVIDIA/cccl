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

// template<bool OtherConst>
//   requires sized_sentinel_for<sentinel_t<Base>, iterator_t<maybe-const<OtherConst, V>>>
// friend constexpr range_difference_t<maybe-const<OtherConst, V>>
//   operator-(const iterator<OtherConst>& x, const sentinel& y);
//
// template<bool OtherConst>
//   requires sized_sentinel_for<sentinel_t<Base>, iterator_t<maybe-const<OtherConst, V>>>
// friend constexpr range_difference_t<maybe-const<OtherConst, V>>
//   operator-(const sentinel& x, const iterator<OtherConst>& y);

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/functional>
#include <cuda/std/ranges>
#include <cuda/std/tuple>

#include "../types.h"
#include "test_macros.h"

template <class T>
void print() = delete;

template <bool Const>
struct Iter
{
  cuda::std::tuple<int>* it_;

  using value_type       = cuda::std::tuple<int>;
  using difference_type  = intptr_t;
  using iterator_concept = cuda::std::input_iterator_tag;

  __host__ __device__ constexpr decltype(auto) operator*() const
  {
    return *it_;
  }
  __host__ __device__ constexpr Iter& operator++()
  {
    ++it_;
    return *this;
  }
  __host__ __device__ constexpr void operator++(int)
  {
    ++it_;
  }
};

template <bool Const>
struct Sent
{
  cuda::std::tuple<int>* end_;

  __host__ __device__ friend constexpr bool operator==(const Sent& s, const Iter<Const>& i)
  {
    return i.it_ == s.end_;
  }
#if TEST_STD_VER <= 2017
  __host__ __device__ friend constexpr bool operator==(const Iter<Const>& i, const Sent& s)
  {
    return i.it_ == s.end_;
  }
  __host__ __device__ friend constexpr bool operator!=(const Sent& s, const Iter<Const>& i)
  {
    return i.it_ != s.end_;
  }
  __host__ __device__ friend constexpr bool operator!=(const Iter<Const>& i, const Sent& s)
  {
    return i.it_ != s.end_;
  }
#endif // TEST_STD_VER <= 2017
};

template <bool Const>
struct SizedSent
{
  cuda::std::tuple<int>* end_;

  __host__ __device__ friend constexpr bool operator==(const SizedSent& s, const Iter<Const>& i)
  {
    return i.it_ == s.end_;
  }
#if TEST_STD_VER <= 2017
  __host__ __device__ friend constexpr bool operator==(const Iter<Const>& i, const SizedSent& s)
  {
    return i.it_ == s.end_;
  }
  __host__ __device__ friend constexpr bool operator!=(const SizedSent& s, const Iter<Const>& i)
  {
    return i.it_ != s.end_;
  }
  __host__ __device__ friend constexpr bool operator!=(const Iter<Const>& i, const SizedSent& s)
  {
    return i.it_ != s.end_;
  }
#endif // TEST_STD_VER <= 2017
  __host__ __device__ friend constexpr intptr_t operator-(const SizedSent& st, const Iter<Const>& it)
  {
    return st.end_ - it.it_;
  }
  __host__ __device__ friend constexpr intptr_t operator-(const Iter<Const>& it, const SizedSent& st)
  {
    return it.it_ - st.end_;
  }
};

template <bool Const>
struct CrossSizedSent
{
  cuda::std::tuple<int>* end_;

  template <bool C>
  __host__ __device__ friend constexpr bool operator==(const CrossSizedSent& s, const Iter<C>& i)
  {
    return i.it_ == s.end_;
  }
#if TEST_STD_VER <= 2017
  template <bool C>
  __host__ __device__ friend constexpr bool operator==(const Iter<C>& i, const CrossSizedSent& s)
  {
    return i.it_ == s.end_;
  }
  template <bool C>
  __host__ __device__ friend constexpr bool operator!=(const CrossSizedSent& s, const Iter<C>& i)
  {
    return i.it_ != s.end_;
  }
  template <bool C>
  __host__ __device__ friend constexpr bool operator!=(const Iter<C>& i, const CrossSizedSent& s)
  {
    return i.it_ != s.end_;
  }
#endif // TEST_STD_VER <= 2017
  template <bool C>
  __host__ __device__ friend constexpr intptr_t operator-(const CrossSizedSent& st, const Iter<C>& it)
  {
    return st.end_ - it.it_;
  }
  template <bool C>
  __host__ __device__ friend constexpr intptr_t operator-(const Iter<C>& it, const CrossSizedSent& st)
  {
    return it.it_ - st.end_;
  }
};

template <template <bool> class It, template <bool> class St>
struct Range : TupleBufferView
{
  DELEGATE_TUPLEBUFFERVIEW(Range)

  using iterator       = It<false>;
  using sentinel       = St<false>;
  using const_iterator = It<true>;
  using const_sentinel = St<true>;

  __host__ __device__ constexpr iterator begin()
  {
    return {buffer_};
  }
  __host__ __device__ constexpr const_iterator begin() const
  {
    return {buffer_};
  }
  __host__ __device__ constexpr sentinel end()
  {
    return sentinel{buffer_ + size_};
  }
  __host__ __device__ constexpr const_sentinel end() const
  {
    return const_sentinel{buffer_ + size_};
  }
};

#if TEST_STD_VER >= 2020
template <class T, class U>
concept HasMinus = requires(const T t, const U u) { t - u; };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class U, class = void>
inline constexpr bool HasMinus = false;

template <class T, class U>
inline constexpr bool
  HasMinus<T, U, cuda::std::void_t<decltype(cuda::std::declval<const T>() - cuda::std::declval<const U>())>> = true;
#endif // TEST_STD_VER <= 2017
template <class BaseRange>
using ElementsView = cuda::std::ranges::elements_view<BaseRange, 0>;

template <class BaseRange>
using ElemIter = cuda::std::ranges::iterator_t<ElementsView<BaseRange>>;

template <class BaseRange>
using ElemConstIter = cuda::std::ranges::iterator_t<const ElementsView<BaseRange>>;

template <class BaseRange>
using ElemSent = cuda::std::ranges::sentinel_t<ElementsView<BaseRange>>;

template <class BaseRange>
using ElemConstSent = cuda::std::ranges::sentinel_t<const ElementsView<BaseRange>>;

__host__ __device__ constexpr void testConstraints()
{
  // base is not sized
  {
    using Base = Range<Iter, Sent>;
    static_assert(!HasMinus<ElemSent<Base>, ElemIter<Base>>);
    static_assert(!HasMinus<ElemIter<Base>, ElemSent<Base>>);

    static_assert(!HasMinus<ElemSent<Base>, ElemConstIter<Base>>);
    static_assert(!HasMinus<ElemConstIter<Base>, ElemSent<Base>>);

    static_assert(!HasMinus<ElemConstSent<Base>, ElemConstIter<Base>>);
    static_assert(!HasMinus<ElemConstIter<Base>, ElemConstSent<Base>>);

    static_assert(!HasMinus<ElemConstSent<Base>, ElemIter<Base>>);
    static_assert(!HasMinus<ElemIter<Base>, ElemConstSent<Base>>);
  }

  // base is sized but not cross const
  {
    using Base = Range<Iter, SizedSent>;
    static_assert(HasMinus<ElemSent<Base>, ElemIter<Base>>);
    static_assert(HasMinus<ElemIter<Base>, ElemSent<Base>>);

    static_assert(!HasMinus<ElemSent<Base>, ElemConstIter<Base>>);
    static_assert(!HasMinus<ElemConstIter<Base>, ElemSent<Base>>);

    static_assert(HasMinus<ElemConstSent<Base>, ElemConstIter<Base>>);
    static_assert(HasMinus<ElemConstIter<Base>, ElemConstSent<Base>>);

    static_assert(!HasMinus<ElemConstSent<Base>, ElemIter<Base>>);
    static_assert(!HasMinus<ElemIter<Base>, ElemConstSent<Base>>);
  }

  // base is cross const sized
  {
    using Base = Range<Iter, CrossSizedSent>;
    static_assert(HasMinus<ElemSent<Base>, ElemIter<Base>>);
    static_assert(HasMinus<ElemIter<Base>, ElemSent<Base>>);

    static_assert(HasMinus<ElemSent<Base>, ElemConstIter<Base>>);
    static_assert(HasMinus<ElemConstIter<Base>, ElemSent<Base>>);

    static_assert(HasMinus<ElemConstSent<Base>, ElemConstIter<Base>>);
    static_assert(HasMinus<ElemConstIter<Base>, ElemConstSent<Base>>);

    static_assert(HasMinus<ElemConstSent<Base>, ElemIter<Base>>);
    static_assert(HasMinus<ElemIter<Base>, ElemConstSent<Base>>);
  }
}

__host__ __device__ constexpr bool test()
{
  cuda::std::tuple<int> buffer[] = {{1}, {2}, {3}, {4}, {5}};

  // base is sized but not cross const
  {
    using Base = Range<Iter, SizedSent>;
    Base base{buffer};
    auto ev         = base | cuda::std::views::elements<0>;
    auto iter       = ev.begin();
    auto const_iter = cuda::std::as_const(ev).begin();
    auto sent       = ev.end();
    auto const_sent = cuda::std::as_const(ev).end();

    assert(iter - sent == -5);
    assert(sent - iter == 5);
    assert(const_iter - const_sent == -5);
    assert(const_sent - const_iter == 5);
  }

  // base is cross const sized
  {
    using Base = Range<Iter, CrossSizedSent>;
    Base base{buffer};
    auto ev         = base | cuda::std::views::elements<0>;
    auto iter       = ev.begin();
    auto const_iter = cuda::std::as_const(ev).begin();
    auto sent       = ev.end();
    auto const_sent = cuda::std::as_const(ev).end();

    assert(iter - sent == -5);
    assert(sent - iter == 5);
    assert(iter - const_sent == -5);
    assert(const_sent - iter == 5);
    assert(const_iter - sent == -5);
    assert(sent - const_iter == 5);
    assert(const_iter - const_sent == -5);
    assert(const_sent - const_iter == 5);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
