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

// cuda::std::views::zip

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"
#include "types.h"

static_assert(cuda::std::is_invocable_v<decltype((cuda::std::views::zip))>);
static_assert(!cuda::std::is_invocable_v<decltype((cuda::std::views::zip)), int>);
static_assert(cuda::std::is_invocable_v<decltype((cuda::std::views::zip)), SizedRandomAccessView>);
static_assert(
  cuda::std::
    is_invocable_v<decltype((cuda::std::views::zip)), SizedRandomAccessView, cuda::std::ranges::iota_view<int, int>>);
static_assert(!cuda::std::is_invocable_v<decltype((cuda::std::views::zip)), SizedRandomAccessView, int>);

__host__ __device__ constexpr bool test()
{
  {
    // zip zero arguments
    auto v = cuda::std::views::zip();
    assert(cuda::std::ranges::empty(v));
    static_assert(cuda::std::is_same_v<decltype(v), cuda::std::ranges::empty_view<cuda::std::tuple<>>>);
  }

  {
    // zip a view
    int buffer[8]    = {1, 2, 3, 4, 5, 6, 7, 8};
    decltype(auto) v = cuda::std::views::zip(SizedRandomAccessView{buffer});
    static_assert(cuda::std::same_as<decltype(v), cuda::std::ranges::zip_view<SizedRandomAccessView>>);
    assert(cuda::std::ranges::size(v) == 8);
    static_assert(cuda::std::is_same_v<cuda::std::ranges::range_reference_t<decltype(v)>, cuda::std::tuple<int&>>);
  }

  {
    // zip a viewable range
    cuda::std::array<int, 3> a{1, 2, 3};
    decltype(auto) v = cuda::std::views::zip(a);
    static_assert(
      cuda::std::same_as<decltype(v),
                         cuda::std::ranges::zip_view<cuda::std::ranges::ref_view<cuda::std::array<int, 3>>>>);
    assert(&(cuda::std::get<0>(*v.begin())) == &(a[0]));
    static_assert(cuda::std::is_same_v<cuda::std::ranges::range_reference_t<decltype(v)>, cuda::std::tuple<int&>>);
  }

  {
    // zip the zip_view
    int buffer[8]    = {1, 2, 3, 4, 5, 6, 7, 8};
    decltype(auto) v = cuda::std::views::zip(SizedRandomAccessView{buffer}, SizedRandomAccessView{buffer});
    static_assert(
      cuda::std::same_as<decltype(v), cuda::std::ranges::zip_view<SizedRandomAccessView, SizedRandomAccessView>>);

    decltype(auto) v2 = cuda::std::views::zip(v);
    static_assert(
      cuda::std::same_as<
        decltype(v2),
        cuda::std::ranges::zip_view<cuda::std::ranges::zip_view<SizedRandomAccessView, SizedRandomAccessView>>>);

    static_assert(cuda::std::is_same_v<cuda::std::ranges::range_reference_t<decltype(v2)>,
                                       cuda::std::tuple<cuda::std::pair<int&, int&>>>);
    unused(v2);
  }
  return true;
}

int main(int, char**)
{
  test();
#if defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // defined(_LIBCUDACXX_ADDRESSOF)

  return 0;
}
