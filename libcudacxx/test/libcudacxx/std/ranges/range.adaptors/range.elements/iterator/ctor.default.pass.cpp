//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// iterator() requires default_initializable<iterator_t<Base>> = default;

#include <cuda/std/ranges>
#include <cuda/std/tuple>

#include "../types.h"
#include "test_macros.h"

struct PODIter : IterBase<PODIter>
{
  int i; // deliberately uninitialised
};

struct IterDefaultCtrView : cuda::std::ranges::view_base
{
  TEST_FUNC PODIter begin() const
  {
    return PODIter{};
  }
  TEST_FUNC PODIter end() const
  {
    return PODIter{};
  }
};

struct IterNoDefaultCtrView : cuda::std::ranges::view_base
{
  TEST_FUNC cpp20_input_iterator<cuda::std::tuple<int>*> begin() const
  {
    return cpp20_input_iterator<cuda::std::tuple<int>*>{nullptr};
  }
  TEST_FUNC sentinel_wrapper<cpp20_input_iterator<cuda::std::tuple<int>*>> end() const
  {
    return sentinel_wrapper<cpp20_input_iterator<cuda::std::tuple<int>*>>{};
  }
};

template <class View, size_t N>
using ElementsIter = cuda::std::ranges::iterator_t<cuda::std::ranges::elements_view<View, N>>;

static_assert(!cuda::std::default_initializable<ElementsIter<IterNoDefaultCtrView, 0>>);
static_assert(cuda::std::default_initializable<ElementsIter<IterDefaultCtrView, 0>>);

TEST_FUNC constexpr bool test()
{
  using Iter = ElementsIter<IterDefaultCtrView, 0>;
  {
    Iter iter;
    assert(iter.base().i == 0); // PODIter has to be initialised to have value 0
  }

  {
    Iter iter = {};
    assert(iter.base().i == 0); // PODIter has to be initialised to have value 0
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
