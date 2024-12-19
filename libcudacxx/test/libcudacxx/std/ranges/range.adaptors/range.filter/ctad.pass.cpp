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

// template <class Range, class Pred>
// filter_view(Range&&, Pred) -> filter_view<views::all_t<Range>, Pred>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"

struct View : cuda::std::ranges::view_base
{
  View() = default;
  __host__ __device__ forward_iterator<int*> begin() const;
  __host__ __device__ sentinel_wrapper<forward_iterator<int*>> end() const;
};
static_assert(cuda::std::ranges::view<View>);

// A range that is not a view
struct Range
{
  Range() = default;
  __host__ __device__ forward_iterator<int*> begin() const;
  __host__ __device__ sentinel_wrapper<forward_iterator<int*>> end() const;
};
static_assert(cuda::std::ranges::range<Range> && !cuda::std::ranges::view<Range>);

struct Pred
{
  __host__ __device__ constexpr bool operator()(int i) const
  {
    return i % 2 == 0;
  }
};

__host__ __device__ constexpr bool test()
{
  {
    View v{};
    Pred pred{};
    cuda::std::ranges::filter_view view(v, pred);
    static_assert(cuda::std::is_same_v<decltype(view), cuda::std::ranges::filter_view<View, Pred>>);
  }

  // Test with a range that isn't a view, to make sure we properly use views::all_t in the implementation.
  {
    Range r{};
    Pred pred{};
    cuda::std::ranges::filter_view view(r, pred);
    static_assert(
      cuda::std::is_same_v<decltype(view), cuda::std::ranges::filter_view<cuda::std::ranges::ref_view<Range>, Pred>>);
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)

  return 0;
}
