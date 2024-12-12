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

//  template<bool OtherConst>
//    requires sentinel_for<sentinel_t<Base>, iterator_t<maybe-const<OtherConst, V>>>
//  friend constexpr bool operator==(const iterator<OtherConst>& x, const sentinel& y);

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "../types.h"
#include "test_macros.h"

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
struct CrossComparableSent
{
  cuda::std::tuple<int>* end_;

  template <bool C>
  __host__ __device__ friend constexpr bool operator==(const CrossComparableSent& s, const Iter<C>& i)
  {
    return i.it_ == s.end_;
  }
#if TEST_STD_VER <= 2017
  template <bool C>
  __host__ __device__ friend constexpr bool operator==(const Iter<C>& i, const CrossComparableSent& s)
  {
    return i.it_ == s.end_;
  }
  template <bool C>
  __host__ __device__ friend constexpr bool operator!=(const CrossComparableSent& s, const Iter<C>& i)
  {
    return i.it_ != s.end_;
  }
  template <bool C>
  __host__ __device__ friend constexpr bool operator!=(const Iter<C>& i, const CrossComparableSent& s)
  {
    return i.it_ != s.end_;
  }
#endif // TEST_STD_VER <= 2017
};

template <template <bool> typename St>
struct Range : TupleBufferView
{
  DELEGATE_TUPLEBUFFERVIEW(Range)
  __host__ __device__ constexpr Iter<false> begin()
  {
    return Iter<false>{buffer_};
  }
  __host__ __device__ constexpr Iter<true> begin() const
  {
    return Iter<true>{buffer_};
  }
  __host__ __device__ constexpr St<false> end()
  {
    return St<false>{buffer_ + size_};
  }
  __host__ __device__ constexpr St<true> end() const
  {
    return St<true>{buffer_ + size_};
  }
};

using R                = Range<Sent>;
using CrossComparableR = Range<CrossComparableSent>;

// Test Constraint
#if TEST_STD_VER >= 2020
template <class I, class S>
concept HasEqual = requires(const I i, const S s) { i == s; };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class I, class S, class = void>
inline constexpr bool HasEqual = false;

template <class I, class S>
inline constexpr bool
  HasEqual<I, S, cuda::std::void_t<decltype(cuda::std::declval<const I>() == cuda::std::declval<const S>())>> = true;
#endif // TEST_STD_VER <= 2017
using cuda::std::ranges::elements_view;
using cuda::std::ranges::iterator_t;
using cuda::std::ranges::sentinel_t;

static_assert(HasEqual<iterator_t<elements_view<R, 0>>, //
                       sentinel_t<elements_view<R, 0>>>);

static_assert(!HasEqual<iterator_t<const elements_view<R, 0>>, //
                        sentinel_t<elements_view<R, 0>>>);

static_assert(!HasEqual<iterator_t<elements_view<R, 0>>, //
                        sentinel_t<const elements_view<R, 0>>>);

static_assert(HasEqual<iterator_t<const elements_view<R, 0>>, //
                       sentinel_t<const elements_view<R, 0>>>);

static_assert(HasEqual<iterator_t<elements_view<R, 0>>, //
                       sentinel_t<elements_view<R, 0>>>);

static_assert(HasEqual<iterator_t<const elements_view<CrossComparableR, 0>>, //
                       sentinel_t<elements_view<CrossComparableR, 0>>>);

static_assert(HasEqual<iterator_t<elements_view<CrossComparableR, 0>>, //
                       sentinel_t<const elements_view<CrossComparableR, 0>>>);

static_assert(HasEqual<iterator_t<const elements_view<CrossComparableR, 0>>, //
                       sentinel_t<const elements_view<CrossComparableR, 0>>>);

template <class R, bool ConstIter, bool ConstSent>
struct getBegin_t
{
  template <class T>
  __host__ __device__ constexpr auto operator()(T&& rng) const
  {
    if constexpr (ConstIter)
    {
      return cuda::std::as_const(rng).begin();
    }
    else
    {
      return rng.begin();
    }
    _CCCL_UNREACHABLE();
  }
};

template <class R, bool ConstIter, bool ConstSent>
struct getEnd_t
{
  template <class T>
  __host__ __device__ constexpr auto operator()(T&& rng) const
  {
    if constexpr (ConstSent)
    {
      return cuda::std::as_const(rng).end();
    }
    else
    {
      return rng.end();
    }
    _CCCL_UNREACHABLE();
  }
};

// cannot declare a non constexpr variable in a constexpr function, so need to pull array out of the main function
template <class R, bool ConstIter, bool ConstSent>
__host__ __device__ TEST_CONSTEXPR_CXX20 void testarray()
{
  getBegin_t<R, ConstIter, ConstSent> getBegin{};
  getEnd_t<R, ConstIter, ConstSent> getEnd{};

  cuda::std::array<cuda::std::tuple<int>, 0> arr;
  R v{arr};
  cuda::std::ranges::elements_view<R, 0> ev(v);
  auto iter = getBegin(ev);
  auto sent = getEnd(ev);
  assert(iter == sent);
}

template <class R, bool ConstIter, bool ConstSent>
__host__ __device__ constexpr void testOne()
{
  // iter == sentinel.base
  getBegin_t<R, ConstIter, ConstSent> getBegin{};
  getEnd_t<R, ConstIter, ConstSent> getEnd{};

  {
    cuda::std::tuple<int> buffer[] = {{1}};
    R v{buffer};
    cuda::std::ranges::elements_view<R, 0> ev(v);
    auto iter = getBegin(ev);
    auto st   = getEnd(ev);
    ++iter;
    assert(iter == st);
  }

  // iter != sentinel.base
  {
    cuda::std::tuple<int> buffer[] = {{1}};
    R v{buffer};
    cuda::std::ranges::elements_view<R, 0> ev(v);
    auto iter = getBegin(ev);
    auto st   = getEnd(ev);
    assert(iter != st);
  }

  // empty range
#if TEST_HAS_BUILTIN(__builtin_is_constant_evaluated)
  if (!__builtin_is_constant_evaluated())
  {
    testarray<R, ConstIter, ConstSent>();
  }
#endif // __builtin_is_constant_evaluated
}

__host__ __device__ constexpr bool test()
{
  testOne<R, false, false>();
  testOne<R, true, true>();
  testOne<CrossComparableR, false, false>();
  testOne<CrossComparableR, true, true>();
  testOne<CrossComparableR, true, false>();
  testOne<CrossComparableR, false, true>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
