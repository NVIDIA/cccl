//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

//  friend constexpr bool operator==(const iterator_t<Base>& x, const sentinel& y);
//
//  template<bool OtherConst = !Const>
//    requires sentinel_for<sentinel_t<Base>, iterator_t<maybe-const<OtherConst, V>>>
//  friend constexpr bool operator==(const iterator_t<maybe-const<OtherConst, V>>& x,
//                                   const sentinel& y);

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "../types.h"
#include "test_macros.h"

template <bool Const>
struct Iter
{
  int* it_;

  using value_type       = int;
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
  int* end_;

  __host__ __device__ friend constexpr bool operator==(const Iter<Const>& i, const Sent& s)
  {
    return i.it_ == s.end_;
  }
#if TEST_STD_VER < 2020
  __host__ __device__ friend constexpr bool operator==(const Sent& s, const Iter<Const>& i)
  {
    return i.it_ == s.end_;
  }
  __host__ __device__ friend constexpr bool operator!=(const Iter<Const>& i, const Sent& s)
  {
    return i.it_ != s.end_;
  }
  __host__ __device__ friend constexpr bool operator!=(const Sent& s, const Iter<Const>& i)
  {
    return i.it_ != s.end_;
  }
#endif // TEST_STD_VER <= 2017
};

template <bool Const>
struct CrossComparableSent
{
  int* end_;

  template <bool C>
  __host__ __device__ friend constexpr bool operator==(const Iter<C>& i, const CrossComparableSent& s)
  {
    return i.it_ == s.end_;
  }
#if TEST_STD_VER < 2020
  template <bool C>
  __host__ __device__ friend constexpr bool operator==(const CrossComparableSent& s, const Iter<C>& i)
  {
    return i.it_ == s.end_;
  }
  template <bool C>
  __host__ __device__ friend constexpr bool operator!=(const Iter<C>& i, const CrossComparableSent& s)
  {
    return i.it_ != s.end_;
  }
  template <bool C>
  __host__ __device__ friend constexpr bool operator!=(const CrossComparableSent& s, const Iter<C>& i)
  {
    return i.it_ != s.end_;
  }
#endif // TEST_STD_VER <= 2017
};

template <template <bool> typename St>
struct Range : IntBufferViewBase
{
#if TEST_COMPILER(NVRTC) // nvbug 3961621
  Range() = default;

  template <class T>
  __host__ __device__ constexpr Range(T&& input)
      : IntBufferViewBase(cuda::std::forward<T>(input))
  {}
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
  // ^^^ TEST_COMPILER(NVRTC) ^^^ / vvv !TEST_COMPILER(NVRTC) vvv
  using IntBufferViewBase::IntBufferViewBase;
#endif // !TEST_COMPILER(NVRTC)

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

struct LessThan3
{
  __host__ __device__ constexpr bool operator()(int i) const
  {
    return i < 3;
  }
};

// Test Constraint
template <class I, class S>
_CCCL_CONCEPT HasEqual = _CCCL_REQUIRES_EXPR((I, S), const I i, const S s)(unused(i == s));

using cuda::std::ranges::iterator_t;
using cuda::std::ranges::sentinel_t;
using cuda::std::ranges::take_while_view;

static_assert(HasEqual<iterator_t<take_while_view<R, LessThan3>>, //
                       sentinel_t<take_while_view<R, LessThan3>>>);

static_assert(!HasEqual<iterator_t<const take_while_view<R, LessThan3>>, //
                        sentinel_t<take_while_view<R, LessThan3>>>);

static_assert(!HasEqual<iterator_t<take_while_view<R, LessThan3>>, //
                        sentinel_t<const take_while_view<R, LessThan3>>>);

static_assert(HasEqual<iterator_t<const take_while_view<R, LessThan3>>, //
                       sentinel_t<const take_while_view<R, LessThan3>>>);

static_assert(HasEqual<iterator_t<take_while_view<R, LessThan3>>, //
                       sentinel_t<take_while_view<R, LessThan3>>>);

static_assert(HasEqual<iterator_t<const take_while_view<CrossComparableR, LessThan3>>, //
                       sentinel_t<take_while_view<CrossComparableR, LessThan3>>>);

static_assert(HasEqual<iterator_t<take_while_view<CrossComparableR, LessThan3>>, //
                       sentinel_t<const take_while_view<CrossComparableR, LessThan3>>>);

static_assert(HasEqual<iterator_t<const take_while_view<CrossComparableR, LessThan3>>, //
                       sentinel_t<const take_while_view<CrossComparableR, LessThan3>>>);

template <bool ConstIter>
struct getBegin
{
  template <class Range>
  __host__ __device__ constexpr auto operator()(Range&& rng) const noexcept
  {
    if constexpr (ConstIter)
    {
      return cuda::std::as_const(rng).begin();
    }
    else
    {
      return rng.begin();
    }
  }
};

template <bool ConstSent>
struct getEnd
{
  template <class Range>
  __host__ __device__ constexpr auto operator()(Range&& rng) const noexcept
  {
    if constexpr (ConstSent)
    {
      return cuda::std::as_const(rng).end();
    }
    else
    {
      return rng.end();
    }
  }
};

// cannot declare a non constexpr variable in a constexpr function, so need to pull array out of the main function
template <class R, bool ConstIter, bool ConstSent>
__host__ __device__ TEST_CONSTEXPR_CXX20 void testarray()
{
  cuda::std::array<int, 0> arr;
  R v{arr};
  cuda::std::ranges::take_while_view twv(v, LessThan3{});
  auto iter = getBegin<ConstIter>{}(twv);
  auto sent = getEnd<ConstSent>{}(twv);
  assert(iter == sent);
}

template <class R, bool ConstIter, bool ConstSent>
__host__ __device__ constexpr void testOne()
{
  // iter == sentinel.base
  {
    int buffer[] = {1};
    R v{buffer};
    cuda::std::ranges::take_while_view twv(v, LessThan3{});
    auto iter = getBegin<ConstIter>{}(twv);
    auto st   = getEnd<ConstSent>{}(twv);
    ++iter;
    assert(iter == st);
  }

  // iter != sentinel.base && pred(*iter)
  {
    int buffer[] = {1, 3, 4};
    R v{buffer};
    cuda::std::ranges::take_while_view twv(v, LessThan3{});
    auto iter = getBegin<ConstIter>{}(twv);
    auto st   = getEnd<ConstSent>{}(twv);
    assert(iter != st);
    ++iter;
    assert(iter == st);
  }

  // iter != sentinel.base && !pred(*iter)
  {
    int buffer[] = {1, 2, 3, 4, 3, 2, 1};
    R v{buffer};
    cuda::std::ranges::take_while_view twv(v, LessThan3{});
    auto iter = getBegin<ConstIter>{}(twv);
    auto sent = getEnd<ConstSent>{}(twv);
    assert(iter != sent);
  }

  // empty range
  if (!cuda::std::__cccl_default_is_constant_evaluated())
  {
    testarray<R, ConstIter, ConstSent>();
  }
}

__host__ __device__ constexpr bool test_invoke()
{ // test cuda::std::invoke is used
  struct Data
  {
    bool b;
  };

  Data buffer[] = {{true}, {true}, {false}};
  cuda::std::ranges::take_while_view twv(buffer, &Data::b);
  auto it = twv.begin();
  auto st = twv.end();
  assert(it != st);

  ++it;
  assert(it != st);

  ++it;
  assert(it == st);

  return true;
}

__host__ __device__ constexpr bool test()
{
  testOne<R, false, false>();
  testOne<R, true, true>();
  testOne<CrossComparableR, false, false>();
  testOne<CrossComparableR, true, true>();

  // LWG 3449 `take_view` and `take_while_view`'s `sentinel<false>` not comparable with their const iterator
  testOne<CrossComparableR, true, false>();
  testOne<CrossComparableR, false, true>();

  return true;
}

int main(int, char**)
{
  test();
#if defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test());
#endif // _CCCL_BUILTIN_ADDRESSOF

  test_invoke();
// GCC complains about access of a constexpr buffer through a runtime variable
#if defined(_CCCL_BUILTIN_ADDRESSOF) && !TEST_COMPILER(GCC)
  static_assert(test_invoke());
#endif // defined(_CCCL_BUILTIN_ADDRESSOF) && !TEST_COMPILER(GCC)

  return 0;
}
