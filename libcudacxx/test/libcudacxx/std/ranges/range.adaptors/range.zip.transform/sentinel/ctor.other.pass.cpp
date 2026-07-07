//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

//  constexpr sentinel(sentinel<!Const> i)
//    requires Const && convertible_to<zentinel<false>, zentinel<Const>>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "../types.h"

template <class T>
struct convertible_sentinel_wrapper
{
  explicit convertible_sentinel_wrapper() = default;
  TEST_FUNC constexpr convertible_sentinel_wrapper(const T& it)
      : it_(it)
  {}

  template <class U, cuda::std::enable_if_t<cuda::std::convertible_to<const U&, T>, int> = 0>
  TEST_FUNC constexpr convertible_sentinel_wrapper(const convertible_sentinel_wrapper<U>& other)
      : it_(other.it_)
  {}

  TEST_FUNC constexpr friend bool operator==(convertible_sentinel_wrapper const& self, const T& other)
  {
    return self.it_ == other;
  }
  T it_;
};

struct NonSimpleNonCommonConvertibleView : IntBufferView
{
  using IntBufferView::IntBufferView;

  TEST_FUNC constexpr int* begin()
  {
    return buffer_;
  }
  TEST_FUNC constexpr const int* begin() const
  {
    return buffer_;
  }
  TEST_FUNC constexpr convertible_sentinel_wrapper<int*> end()
  {
    return convertible_sentinel_wrapper<int*>(buffer_ + size_);
  }
  TEST_FUNC constexpr convertible_sentinel_wrapper<const int*> end() const
  {
    return convertible_sentinel_wrapper<const int*>(buffer_ + size_);
  }
};

// convertible_to<zentinel<false>, zentinel<Const>>
static_assert(cuda::std::convertible_to< //
              cuda::std::ranges::sentinel_t<cuda::std::ranges::zip_view<NonSimpleNonCommonConvertibleView>>,
              cuda::std::ranges::sentinel_t<cuda::std::ranges::zip_view<NonSimpleNonCommonConvertibleView> const>>);

TEST_FUNC constexpr bool test()
{
  int buffer1[4] = {1, 2, 3, 4};
  int buffer2[5] = {1, 2, 3, 4, 5};
  {
    cuda::std::ranges::zip_transform_view v{
      MakeTuple{}, NonSimpleNonCommonConvertibleView(buffer1), NonSimpleNonCommonConvertibleView(buffer2)};
    using ZipTransformView = decltype(v);
    static_assert(!cuda::std::ranges::common_range<ZipTransformView>);
    auto sent1                                                  = v.end();
    cuda::std::ranges::sentinel_t<const ZipTransformView> sent2 = sent1;
    static_assert(!cuda::std::is_same_v<decltype(sent1), decltype(sent2)>);

    assert(v.begin() != sent2);
    assert(cuda::std::as_const(v).begin() != sent2);
    assert(v.begin() + 4 == sent2);
    assert(cuda::std::as_const(v).begin() + 4 == sent2);

    // Cannot create a non-const iterator from a const iterator.
    static_assert(!cuda::std::constructible_from<decltype(sent1), decltype(sent2)>);
  }

  {
    // one range
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, NonSimpleNonCommonConvertibleView{buffer1});
    auto sent1                                             = v.end();
    cuda::std::ranges::sentinel_t<const decltype(v)> sent2 = sent1;
    static_assert(!cuda::std::is_same_v<decltype(sent1), decltype(sent2)>);
    assert(v.begin() != sent1);
    assert(v.begin() != sent2);
    assert(v.begin() + 4 == sent1);
    assert(v.begin() + 4 == sent2);
  }

  {
    // two ranges
    cuda::std::ranges::zip_transform_view v(
      GetFirst{}, NonSimpleNonCommonConvertibleView{buffer1}, cuda::std::views::iota(0));
    auto sent1                                             = v.end();
    cuda::std::ranges::sentinel_t<const decltype(v)> sent2 = sent1;
    static_assert(!cuda::std::is_same_v<decltype(sent1), decltype(sent2)>);
    assert(v.begin() != sent1);
    assert(v.begin() != sent2);
    assert(v.begin() + 4 == sent1);
    assert(v.begin() + 4 == sent2);
  }

  {
    // three ranges
    cuda::std::ranges::zip_transform_view v(
      Tie{}, NonSimpleNonCommonConvertibleView{buffer1}, SimpleCommon{buffer1}, cuda::std::ranges::single_view(2.));
    auto sent1                                             = v.end();
    cuda::std::ranges::sentinel_t<const decltype(v)> sent2 = sent1;
    static_assert(!cuda::std::is_same_v<decltype(sent1), decltype(sent2)>);
    assert(v.begin() != sent1);
    assert(v.begin() != sent2);
    assert(v.begin() + 1 == sent1);
    assert(v.begin() + 1 == sent2);
  }

  {
    // single empty range
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, NonSimpleNonCommonConvertibleView(nullptr, 0));
    auto sent1                                             = v.end();
    cuda::std::ranges::sentinel_t<const decltype(v)> sent2 = sent1;
    static_assert(!cuda::std::is_same_v<decltype(sent1), decltype(sent2)>);
    assert(v.begin() == sent1);
    assert(v.begin() == sent2);
  }

  {
    // empty range at the beginning
    cuda::std::ranges::zip_transform_view v(
      MakeTuple{},
      cuda::std::ranges::empty_view<int>(),
      NonSimpleNonCommonConvertibleView{buffer1},
      SimpleCommon{buffer1});
    auto sent1                                             = v.end();
    cuda::std::ranges::sentinel_t<const decltype(v)> sent2 = sent1;
    static_assert(!cuda::std::is_same_v<decltype(sent1), decltype(sent2)>);
    assert(v.begin() == sent1);
    assert(v.begin() == sent2);
  }

  {
    // empty range in the middle
    cuda::std::ranges::zip_transform_view v(
      MakeTuple{},
      SimpleCommon{buffer1},
      cuda::std::ranges::empty_view<int>(),
      NonSimpleNonCommonConvertibleView{buffer1});
    auto sent1                                             = v.end();
    cuda::std::ranges::sentinel_t<const decltype(v)> sent2 = sent1;
    static_assert(!cuda::std::is_same_v<decltype(sent1), decltype(sent2)>);
    assert(v.begin() == sent1);
    assert(v.begin() == sent2);
  }

  {
    // empty range at the end
    cuda::std::ranges::zip_transform_view v(
      MakeTuple{},
      SimpleCommon{buffer1},
      NonSimpleNonCommonConvertibleView{buffer1},
      cuda::std::ranges::empty_view<int>());
    auto sent1                                             = v.end();
    cuda::std::ranges::sentinel_t<const decltype(v)> sent2 = sent1;
    static_assert(!cuda::std::is_same_v<decltype(sent1), decltype(sent2)>);
    assert(v.begin() == sent1);
    assert(v.begin() == sent2);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
