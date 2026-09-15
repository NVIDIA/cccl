//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr iterator(iterator<!Const> i)
//     requires Const && convertible_to<ziperator<false>, ziperator<Const>>;

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "../types.h"

using ConstIterIncompatibleView =
  BasicView<forward_iterator<int*>,
            forward_iterator<int*>,
            random_access_iterator<const int*>,
            random_access_iterator<const int*>>;
static_assert(!cuda::std::convertible_to<cuda::std::ranges::iterator_t<ConstIterIncompatibleView>,
                                         cuda::std::ranges::iterator_t<const ConstIterIncompatibleView>>);

TEST_FUNC constexpr bool test()
{
  int buffer[3] = {1, 2, 3};

  {
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, NonSimpleCommon{buffer});
    auto iter1                                             = v.begin();
    cuda::std::ranges::iterator_t<const decltype(v)> iter2 = iter1;
    assert(iter1 == iter2);

    static_assert(!cuda::std::is_same_v<decltype(iter1), decltype(iter2)>);

    // We cannot create a non-const iterator from a const iterator.
    static_assert(!cuda::std::constructible_from<decltype(iter1), decltype(iter2)>);
  }

  {
    // Check when we can't perform a non-const-to-const conversion of the ziperator
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, ConstIterIncompatibleView{buffer});
    auto iter1 = v.begin();
    auto iter2 = cuda::std::as_const(v).begin();

    static_assert(!cuda::std::is_same_v<decltype(iter1), decltype(iter2)>);

    static_assert(!cuda::std::constructible_from<decltype(iter1), decltype(iter2)>);
    static_assert(!cuda::std::constructible_from<decltype(iter2), decltype(iter1)>);
  }

  {
    // one range
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, NonSimpleCommon{buffer});
    auto iter1                                             = v.begin();
    cuda::std::ranges::iterator_t<const decltype(v)> iter2 = iter1;
    static_assert(!cuda::std::is_same_v<decltype(iter1), decltype(iter2)>);
    assert(*iter2 == cuda::std::tuple(1));
  }

  {
    // two ranges
    cuda::std::ranges::zip_transform_view v(GetFirst{}, NonSimpleCommon{buffer}, cuda::std::views::iota(0));
    auto iter1                                             = v.begin();
    cuda::std::ranges::iterator_t<const decltype(v)> iter2 = iter1;
    static_assert(!cuda::std::is_same_v<decltype(iter1), decltype(iter2)>);
    assert(*iter2 == 1);
  }

  {
    // three ranges
    cuda::std::ranges::zip_transform_view v(
      Tie{}, NonSimpleCommon{buffer}, SimpleCommon{buffer}, cuda::std::ranges::single_view(2.));
    auto iter1                                             = v.begin();
    cuda::std::ranges::iterator_t<const decltype(v)> iter2 = iter1;
    static_assert(!cuda::std::is_same_v<decltype(iter1), decltype(iter2)>);
    assert(*iter2 == cuda::std::tuple(1, 1, 2.0));
  }

  {
    // single empty range
    cuda::std::array<int, 0> buffer2{};
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, buffer2);
    auto iter1                                             = v.begin();
    cuda::std::ranges::iterator_t<const decltype(v)> iter2 = iter1;
    static_assert(!cuda::std::is_same_v<decltype(iter1), decltype(iter2)>);
    assert(iter2 == v.end());
  }

  {
    // empty range at the beginning
    cuda::std::ranges::zip_transform_view v(
      MakeTuple{}, cuda::std::ranges::empty_view<int>(), NonSimpleCommon{buffer}, SimpleCommon{buffer});
    auto iter1                                             = v.begin();
    cuda::std::ranges::iterator_t<const decltype(v)> iter2 = iter1;
    static_assert(!cuda::std::is_same_v<decltype(iter1), decltype(iter2)>);
    assert(iter2 == v.end());
  }

  {
    // empty range in the middle
    cuda::std::ranges::zip_transform_view v(
      MakeTuple{}, SimpleCommon{buffer}, cuda::std::ranges::empty_view<int>(), NonSimpleCommon{buffer});
    auto iter1                                             = v.begin();
    cuda::std::ranges::iterator_t<const decltype(v)> iter2 = iter1;
    static_assert(!cuda::std::is_same_v<decltype(iter1), decltype(iter2)>);
    assert(iter2 == v.end());
  }

  {
    // empty range at the end
    cuda::std::ranges::zip_transform_view v(
      MakeTuple{}, SimpleCommon{buffer}, NonSimpleCommon{buffer}, cuda::std::ranges::empty_view<int>());
    auto iter1                                             = v.begin();
    cuda::std::ranges::iterator_t<const decltype(v)> iter2 = iter1;
    static_assert(!cuda::std::is_same_v<decltype(iter1), decltype(iter2)>);
    assert(iter2 == v.end());
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
