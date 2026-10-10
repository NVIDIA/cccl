//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// iterator() = default;

#include <cuda/std/ranges>
#include <cuda/std/tuple>

#include "../types.h"
#include "test_macros.h"

struct PODIter
{
  int i; // deliberately uninitialised

  using iterator_category = cuda::std::random_access_iterator_tag;
  using value_type        = int;
  using difference_type   = intptr_t;

  TEST_FUNC constexpr int operator*() const
  {
    return i;
  }

  TEST_FUNC constexpr PODIter& operator++()
  {
    return *this;
  }
  TEST_FUNC constexpr void operator++(int) {}

#if TEST_STD_VER >= 2020
  TEST_FUNC friend constexpr bool operator==(const PODIter&, const PODIter&) = default;
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
  TEST_FUNC friend constexpr bool operator==(const PODIter& lhs, const PODIter& rhs)
  {
    return lhs.i == rhs.i;
  }
  TEST_FUNC friend constexpr bool operator!=(const PODIter& lhs, const PODIter& rhs)
  {
    return lhs.i != rhs.i;
  }
#endif // TEST_STD_VER <=2017
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
  TEST_FUNC cpp20_input_iterator<int*> begin() const
  {
    return cpp20_input_iterator<int*>{nullptr};
  }
  TEST_FUNC sentinel_wrapper<cpp20_input_iterator<int*>> end() const
  {
    return sentinel_wrapper<cpp20_input_iterator<int*>>{};
  }
};

template <class... Views>
using zip_iter = cuda::std::ranges::iterator_t<cuda::std::ranges::zip_view<Views...>>;

static_assert(!cuda::std::default_initializable<zip_iter<IterNoDefaultCtrView>>);
static_assert(!cuda::std::default_initializable<zip_iter<IterNoDefaultCtrView, IterDefaultCtrView>>);
static_assert(!cuda::std::default_initializable<zip_iter<IterNoDefaultCtrView, IterNoDefaultCtrView>>);
static_assert(cuda::std::default_initializable<zip_iter<IterDefaultCtrView>>);
static_assert(cuda::std::default_initializable<zip_iter<IterDefaultCtrView, IterDefaultCtrView>>);

TEST_FUNC constexpr bool test()
{
  using ZipIter = zip_iter<IterDefaultCtrView>;
  {
    ZipIter iter;
    auto [x] = *iter;
    assert(x == 0); // PODIter has to be initialised to have value 0
  }

  {
    ZipIter iter = {};
    auto [x]     = *iter;
    assert(x == 0); // PODIter has to be initialised to have value 0
  }
  return true;
}

int main(int, char**)
{
  test();
#if defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test());
#endif // defined(_CCCL_BUILTIN_ADDRESSOF)

  return 0;
}
