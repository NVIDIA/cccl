//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// Older Clangs don't properly deduce decltype(auto) with a concept constraint
// XFAIL: apple-clang-13.0

// constexpr Pred const& pred() const;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>

#include "test_macros.h"

struct Range : cuda::std::ranges::view_base
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

struct Pred
{
  __host__ __device__ bool operator()(int) const;
  int value;
};

__host__ __device__ constexpr bool test()
{
  {
    Pred pred{42};
    cuda::std::ranges::filter_view<Range, Pred> const view(Range{}, pred);
    decltype(auto) result = view.pred();
    static_assert(cuda::std::same_as<decltype(result), Pred const&>);
    assert(result.value == 42);

    // Make sure we're really holding a reference to something inside the view
    decltype(auto) result2 = view.pred();
    static_assert(cuda::std::same_as<decltype(result2), Pred const&>);
    assert(&result == &result2);
  }

  // Same, but calling on a non-const view
  {
    Pred pred{42};
    cuda::std::ranges::filter_view<Range, Pred> view(Range{}, pred);
    decltype(auto) result = view.pred();
    static_assert(cuda::std::same_as<decltype(result), Pred const&>);
    assert(result.value == 42);

    // Make sure we're really holding a reference to something inside the view
    decltype(auto) result2 = view.pred();
    static_assert(cuda::std::same_as<decltype(result2), Pred const&>);
    assert(&result == &result2);
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2020

  return 0;
}
