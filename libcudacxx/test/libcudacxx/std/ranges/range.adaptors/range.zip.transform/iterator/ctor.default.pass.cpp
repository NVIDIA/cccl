//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// iterator() = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "../types.h"

struct IterDefaultCtrView : cuda::std::ranges::view_base
{
  TEST_FUNC int* begin() const;
  TEST_FUNC int* end() const;
};

struct IterNoDefaultCtrView : cuda::std::ranges::view_base
{
  TEST_FUNC cpp20_input_iterator<int*> begin() const;
  TEST_FUNC sentinel_wrapper<cpp20_input_iterator<int*>> end() const;
};

template <class... Views>
using Iter = cuda::std::ranges::iterator_t<cuda::std::ranges::zip_transform_view<MakeTuple, Views...>>;

static_assert(!cuda::std::default_initializable<Iter<IterNoDefaultCtrView>>);
static_assert(!cuda::std::default_initializable<Iter<IterNoDefaultCtrView, IterDefaultCtrView>>);
static_assert(!cuda::std::default_initializable<Iter<IterNoDefaultCtrView, IterNoDefaultCtrView>>);
static_assert(cuda::std::default_initializable<Iter<IterDefaultCtrView>>);
static_assert(cuda::std::default_initializable<Iter<IterDefaultCtrView, IterDefaultCtrView>>);

template <class Fn, class... Views>
TEST_FUNC constexpr void test()
{
  using ZipTransformIter = cuda::std::ranges::iterator_t<cuda::std::ranges::zip_transform_view<Fn, Views...>>;
  ZipTransformIter iter1 = {};
  ZipTransformIter iter2;
  assert(iter1 == iter2);
}

TEST_FUNC constexpr bool test()
{
  test<MakeTuple, IterDefaultCtrView>();
  test<MakeTuple, IterDefaultCtrView, cuda::std::ranges::empty_view<int>>();
  test<MakeTuple, IterDefaultCtrView, cuda::std::ranges::iota_view<int>, cuda::std::ranges::single_view<int>>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
