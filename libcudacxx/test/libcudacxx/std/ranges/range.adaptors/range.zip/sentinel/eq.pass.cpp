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
//   requires sentinel_for<sentinel_t<Base>, iterator_t<maybe-const<OtherConst, V>>>
// friend constexpr bool operator==(const iterator<OtherConst>& x, const sentinel& y);

#include <cuda/std/cassert>
#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#  include <cuda/std/compare>
#endif // !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#include <cuda/std/ranges>
#include <cuda/std/tuple>

#include "../types.h"
#include "test_macros.h"

using Iterator      = random_access_iterator<int*>;
using ConstIterator = random_access_iterator<const int*>;

template <bool Const>
struct ComparableSentinel
{
  using Iter = cuda::std::conditional_t<Const, ConstIterator, Iterator>;
  Iter iter_;

  explicit ComparableSentinel() = default;
  __host__ __device__ constexpr explicit ComparableSentinel(const Iter& it)
      : iter_(it)
  {}

  __host__ __device__ constexpr friend bool operator==(const Iterator& i, const ComparableSentinel& s)
  {
    return base(i) == base(s.iter_);
  }
#if TEST_STD_VER <= 2017
  __host__ __device__ constexpr friend bool operator==(const ComparableSentinel& s, const Iterator& i)
  {
    return base(i) == base(s.iter_);
  }
  __host__ __device__ constexpr friend bool operator!=(const Iterator& i, const ComparableSentinel& s)
  {
    return base(i) != base(s.iter_);
  }
  __host__ __device__ constexpr friend bool operator!=(const ComparableSentinel& s, const Iterator& i)
  {
    return base(i) != base(s.iter_);
  }
#endif // TEST_STD_VER <= 2017

  __host__ __device__ constexpr friend bool operator==(const ConstIterator& i, const ComparableSentinel& s)
  {
    return base(i) == base(s.iter_);
  }
#if TEST_STD_VER <= 2017
  __host__ __device__ constexpr friend bool operator==(const ComparableSentinel& s, const ConstIterator& i)
  {
    return base(i) == base(s.iter_);
  }
  __host__ __device__ constexpr friend bool operator!=(const ConstIterator& i, const ComparableSentinel& s)
  {
    return base(i) != base(s.iter_);
  }
  __host__ __device__ constexpr friend bool operator!=(const ComparableSentinel& s, const ConstIterator& i)
  {
    return base(i) != base(s.iter_);
  }
#endif // TEST_STD_VER <= 2017
};

struct ComparableView : IntBufferView
{
#if defined(TEST_COMPILER_NVRTC) // nvbug 3961621
  template <cuda::std::size_t N>
  __host__ __device__ constexpr ComparableView(int (&b)[N])
      : IntBufferView(b)
  {}
#else
  using IntBufferView::IntBufferView;
#endif

  __host__ __device__ constexpr auto begin()
  {
    return Iterator(buffer_);
  }
  __host__ __device__ constexpr auto begin() const
  {
    return ConstIterator(buffer_);
  }
  __host__ __device__ constexpr auto end()
  {
    return ComparableSentinel<false>(Iterator(buffer_ + size_));
  }
  __host__ __device__ constexpr auto end() const
  {
    return ComparableSentinel<true>(ConstIterator(buffer_ + size_));
  }
};

struct ConstIncompatibleView : cuda::std::ranges::view_base
{
  __host__ __device__ cpp17_input_iterator<int*> begin()
  {
    return cpp17_input_iterator<int*>{nullptr};
  }
  __host__ __device__ forward_iterator<const int*> begin() const
  {
    return forward_iterator<const int*>{nullptr};
  }
  __host__ __device__ sentinel_wrapper<cpp17_input_iterator<int*>> end()
  {
    return sentinel_wrapper<cpp17_input_iterator<int*>>{};
  }
  __host__ __device__ sentinel_wrapper<forward_iterator<const int*>> end() const
  {
    return sentinel_wrapper<forward_iterator<const int*>>{};
  }
};

// clang-format off
template <class Iter, class Sent>
inline constexpr bool EqualComparable = cuda::std::invocable<cuda::std::equal_to<>, const Iter&, const Sent&>;
// clang-format on

__host__ __device__ constexpr bool test()
{
  int buffer1[4] = {1, 2, 3, 4};
  int buffer2[5] = {1, 2, 3, 4, 5};
  int buffer3[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  {
    // simple-view: const and non-const have the same iterator/sentinel type
    cuda::std::ranges::zip_view v{SimpleNonCommon(buffer1), SimpleNonCommon(buffer2), SimpleNonCommon(buffer3)};
    static_assert(!cuda::std::ranges::common_range<decltype(v)>);
    static_assert(simple_view<decltype(v)>);

    assert(v.begin() != v.end());
    assert(v.begin() + 1 != v.end());
    assert(v.begin() + 2 != v.end());
    assert(v.begin() + 3 != v.end());
    assert(v.begin() + 4 == v.end());
  }

  {
    // !simple-view: const and non-const have different iterator/sentinel types
    cuda::std::ranges::zip_view v{NonSimpleNonCommon(buffer1), SimpleNonCommon(buffer2), SimpleNonCommon(buffer3)};
    static_assert(!cuda::std::ranges::common_range<decltype(v)>);
    static_assert(!simple_view<decltype(v)>);

    assert(v.begin() != v.end());
    assert(v.begin() + 4 == v.end());

    // const_iterator (const int*) converted to iterator (int*)
#if defined(TEST_COMPILER_NVRTC) // conversion from "int *" to "const int *" is invalid in constant-expression
    if (!cuda::std::is_constant_evaluated())
#endif // TEST_COMPILER_NVRTC
      assert(v.begin() + 4 == cuda::std::as_const(v).end());

    using Iter      = cuda::std::ranges::iterator_t<decltype(v)>;
    using ConstIter = cuda::std::ranges::iterator_t<const decltype(v)>;
    static_assert(!cuda::std::is_same_v<Iter, ConstIter>);
    using Sentinel      = cuda::std::ranges::sentinel_t<decltype(v)>;
    using ConstSentinel = cuda::std::ranges::sentinel_t<const decltype(v)>;
    static_assert(!cuda::std::is_same_v<Sentinel, ConstSentinel>);

    static_assert(EqualComparable<Iter, Sentinel>);
    static_assert(!EqualComparable<ConstIter, Sentinel>);
    static_assert(EqualComparable<Iter, ConstSentinel>);
    static_assert(EqualComparable<ConstIter, ConstSentinel>);
  }

  {
    // underlying const/non-const sentinel can be compared with both const/non-const iterator
    cuda::std::ranges::zip_view v{ComparableView(buffer1), ComparableView(buffer2)};
    static_assert(!cuda::std::ranges::common_range<decltype(v)>);
    static_assert(!simple_view<decltype(v)>);

    assert(v.begin() != v.end());
    assert(v.begin() + 4 == v.end());
    assert(cuda::std::as_const(v).begin() + 4 == v.end());
    assert(cuda::std::as_const(v).begin() + 4 == cuda::std::as_const(v).end());
    assert(v.begin() + 4 == cuda::std::as_const(v).end());

    using Iter      = cuda::std::ranges::iterator_t<decltype(v)>;
    using ConstIter = cuda::std::ranges::iterator_t<const decltype(v)>;
    static_assert(!cuda::std::is_same_v<Iter, ConstIter>);
    using Sentinel      = cuda::std::ranges::sentinel_t<decltype(v)>;
    using ConstSentinel = cuda::std::ranges::sentinel_t<const decltype(v)>;
    static_assert(!cuda::std::is_same_v<Sentinel, ConstSentinel>);

    static_assert(EqualComparable<Iter, Sentinel>);
    static_assert(EqualComparable<ConstIter, Sentinel>);
    static_assert(EqualComparable<Iter, ConstSentinel>);
    static_assert(EqualComparable<ConstIter, ConstSentinel>);
  }

  {
    // underlying const/non-const sentinel cannot be compared with non-const/const iterator
    cuda::std::ranges::zip_view v{ComparableView(buffer1), ConstIncompatibleView{}};
    static_assert(!cuda::std::ranges::common_range<decltype(v)>);
    static_assert(!simple_view<decltype(v)>);

    using Iter      = cuda::std::ranges::iterator_t<decltype(v)>;
    using ConstIter = cuda::std::ranges::iterator_t<const decltype(v)>;
    static_assert(!cuda::std::is_same_v<Iter, ConstIter>);
    using Sentinel      = cuda::std::ranges::sentinel_t<decltype(v)>;
    using ConstSentinel = cuda::std::ranges::sentinel_t<const decltype(v)>;
    static_assert(!cuda::std::is_same_v<Sentinel, ConstSentinel>);

    static_assert(EqualComparable<Iter, Sentinel>);
    static_assert(!EqualComparable<ConstIter, Sentinel>);
    static_assert(!EqualComparable<Iter, ConstSentinel>);
    static_assert(EqualComparable<ConstIter, ConstSentinel>);
  }
  return true;
}

int main(int, char**)
{
  test();
#if defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // _LIBCUDACXX_ADDRESSOF

  return 0;
}
