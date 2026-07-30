//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<bool OtherConst>
//   requires sentinel_for<zentinel<Const>, ziperator<OtherConst>>
// friend constexpr bool operator==(const iterator<OtherConst>& x, const sentinel& y);

#include <cuda/std/cassert>
#include <cuda/std/compare>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "../types.h"

using Iterator      = random_access_iterator<int*>;
using ConstIterator = random_access_iterator<const int*>;

template <bool Const>
struct ComparableSentinel
{
  using Iter = cuda::std::conditional_t<Const, ConstIterator, Iterator>;
  Iter iter_;

  explicit ComparableSentinel() = default;
  TEST_FUNC constexpr explicit ComparableSentinel(const Iter& it)
      : iter_(it)
  {}

  TEST_FUNC constexpr friend bool operator==(const Iterator& i, const ComparableSentinel& s)
  {
    return base(i) == base(s.iter_);
  }

  TEST_FUNC constexpr friend bool operator==(const ConstIterator& i, const ComparableSentinel& s)
  {
    return base(i) == base(s.iter_);
  }
};

struct ComparableView : IntBufferView
{
  using IntBufferView::IntBufferView;

  TEST_FUNC constexpr auto begin()
  {
    return Iterator(buffer_);
  }
  TEST_FUNC constexpr auto begin() const
  {
    return ConstIterator(buffer_);
  }
  TEST_FUNC constexpr auto end()
  {
    return ComparableSentinel<false>(Iterator(buffer_ + size_));
  }
  TEST_FUNC constexpr auto end() const
  {
    return ComparableSentinel<true>(ConstIterator(buffer_ + size_));
  }
};

struct ConstIncompatibleView : cuda::std::ranges::view_base
{
  TEST_FUNC cpp17_input_iterator<int*> begin();
  TEST_FUNC forward_iterator<const int*> begin() const;
  TEST_FUNC sentinel_wrapper<cpp17_input_iterator<int*>> end();
  TEST_FUNC sentinel_wrapper<forward_iterator<const int*>> end() const;
};

template <class Iter, class Sent>
_CCCL_CONCEPT EqualComparable = cuda::std::invocable<cuda::std::equal_to<>, const Iter&, const Sent&>;

TEST_FUNC constexpr bool test()
{
  int buffer1[4] = {1, 2, 3, 4};
  int buffer2[5] = {1, 2, 3, 4, 5};
  int buffer3[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  {
    // const and non-const have different iterator/sentinel types
    cuda::std::ranges::zip_transform_view v{
      MakeTuple{}, NonSimpleNonCommon(buffer1), SimpleNonCommon(buffer2), SimpleNonCommon(buffer3)};
    using ZipTransformView = decltype(v);
    static_assert(!cuda::std::ranges::common_range<ZipTransformView>);
    static_assert(!simple_view<ZipTransformView>);

    assert(v.begin() != v.end());
    assert(v.begin() + 4 == v.end());

    // const_iterator (const int*) converted to iterator (int*)
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
    cuda::std::ranges::zip_transform_view v{MakeTuple{}, ComparableView(buffer1), ComparableView(buffer2)};
    using ZipTransformView = decltype(v);
    static_assert(!cuda::std::ranges::common_range<ZipTransformView>);
    static_assert(!simple_view<ZipTransformView>);

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
    cuda::std::ranges::zip_transform_view v{MakeTuple{}, ComparableView(buffer1), ConstIncompatibleView{}};
    using ZipTransformView = decltype(v);
    static_assert(!cuda::std::ranges::common_range<ZipTransformView>);
    static_assert(!simple_view<ZipTransformView>);

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

  {
    // one range
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, ComparableView{buffer1});
    assert(v.begin() != v.end());
    assert(v.begin() + 4 == v.end());
    assert(v.begin() + 4 == cuda::std::as_const(v).end());
  }

  {
    // two ranges
    cuda::std::ranges::zip_transform_view v(GetFirst{}, ComparableView{buffer2}, cuda::std::views::iota(0));
    assert(v.begin() != v.end());
    assert(v.begin() + 5 == v.end());
    assert(v.begin() + 5 == cuda::std::as_const(v).end());
  }

  {
    // three ranges
    cuda::std::ranges::zip_transform_view v(
      Tie{}, ComparableView{buffer1}, SimpleNonCommon{buffer2}, cuda::std::ranges::single_view(2.));
    assert(v.begin() != v.end());
    assert(v.begin() + 1 == v.end());
    assert(v.begin() + 1 == cuda::std::as_const(v).end());
  }

  {
    // single empty range
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, ComparableView(nullptr, 0));
    assert(v.begin() == v.end());
    assert(cuda::std::as_const(v).begin() == cuda::std::as_const(v).end());
  }

  {
    // empty range at the beginning
    cuda::std::ranges::zip_transform_view v(
      MakeTuple{}, cuda::std::ranges::empty_view<int>(), ComparableView{buffer1}, SimpleCommon{buffer2});
    assert(v.begin() == v.end());
    assert(cuda::std::as_const(v).begin() == cuda::std::as_const(v).end());
  }

  {
    // empty range in the middle
    cuda::std::ranges::zip_transform_view v(
      MakeTuple{}, SimpleCommon{buffer1}, cuda::std::ranges::empty_view<int>(), ComparableView{buffer2});
    assert(v.begin() == v.end());
    assert(cuda::std::as_const(v).begin() == cuda::std::as_const(v).end());
  }

  {
    // empty range at the end
    cuda::std::ranges::zip_transform_view v(
      MakeTuple{}, SimpleCommon{buffer1}, ComparableView{buffer2}, cuda::std::ranges::empty_view<int>());
    assert(v.begin() == v.end());
    assert(cuda::std::as_const(v).begin() == cuda::std::as_const(v).end());
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
