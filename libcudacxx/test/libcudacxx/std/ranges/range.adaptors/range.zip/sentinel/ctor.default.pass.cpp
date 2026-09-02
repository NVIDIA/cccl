//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// sentinel() = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/tuple>

#include "test_macros.h"

struct PODSentinel
{
  bool b; // deliberately uninitialised

  TEST_FUNC friend constexpr bool operator==(int*, const PODSentinel& s)
  {
    return s.b;
  }
#if TEST_STD_VER <= 2017
  TEST_FUNC friend constexpr bool operator==(const PODSentinel& s, int*)
  {
    return s.b;
  }
  TEST_FUNC friend constexpr bool operator!=(int*, const PODSentinel& s)
  {
    return !s.b;
  }
  TEST_FUNC friend constexpr bool operator!=(const PODSentinel& s, int*)
  {
    return !s.b;
  }
#endif // TEST_STD_VER <= 2017
};

struct Range : cuda::std::ranges::view_base
{
  TEST_FUNC int* begin() const
  {
    return nullptr;
  }
  TEST_FUNC PODSentinel end()
  {
    return PODSentinel{};
  }
};

TEST_FUNC constexpr bool test()
{
  {
    using R        = cuda::std::ranges::zip_view<Range>;
    using Sentinel = cuda::std::ranges::sentinel_t<R>;
    static_assert(!cuda::std::is_same_v<Sentinel, cuda::std::ranges::iterator_t<R>>);

    cuda::std::ranges::iterator_t<R> it;

    Sentinel s1;
    assert(it != s1); // PODSentinel.b is initialised to false

    Sentinel s2 = {};
    assert(it != s2); // PODSentinel.b is initialised to false
  }
  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test());
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)

  return 0;
}
