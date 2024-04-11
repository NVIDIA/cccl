//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11
// UNSUPPORTED: msvc && c++14, msvc && c++17

#include <cuda/std/cassert>
#include <cuda/std/mdspan>

int main(int, char**)
{
  // TEST(TestSubmdspanLayoutRightStaticSizedRankReducing3Dto1D,
  // test_submdspan_layout_right_static_sized_rank_reducing_3d_to_1d)
  {
    cuda::std::array<int, 2 * 3 * 4> d;
    cuda::std::mdspan<int, cuda::std::extents<size_t, 2, 3, 4>> m(d.data());
    m(1, 1, 1) = 42;
    auto sub0  = cuda::std::submdspan(m, 1, 1, cuda::std::full_extent);

    static_assert(decltype(sub0)::rank() == 1, "unexpected submdspan rank");
    static_assert(sub0.rank() == 1, "");
    static_assert(sub0.rank_dynamic() == 0, "");
    assert(sub0.extent(0) == 4);
    assert(sub0(1) == 42);
  }

  // TEST(TestSubmdspanLayoutLeftStaticSizedRankReducing3Dto1D,
  // test_submdspan_layout_left_static_sized_rank_reducing_3d_to_1d)
  {
    cuda::std::array<int, 2 * 3 * 4> d;
    cuda::std::mdspan<int, cuda::std::extents<size_t, 2, 3, 4>, cuda::std::layout_left> m(d.data());
    m(1, 1, 1) = 42;
    auto sub0  = cuda::std::submdspan(m, 1, 1, cuda::std::full_extent);

    static_assert(sub0.rank() == 1, "");
    static_assert(sub0.rank_dynamic() == 0, "");
    assert(sub0.extent(0) == 4);
    assert(sub0(1) == 42);
  }

  // TEST(TestSubmdspanLayoutRightStaticSizedRankReducingNested3Dto0D,
  // test_submdspan_layout_right_static_sized_rank_reducing_nested_3d_to_0d)
  {
    cuda::std::array<int, 2 * 3 * 4> d;
    cuda::std::mdspan<int, cuda::std::extents<size_t, 2, 3, 4>> m(d.data());
    m(1, 1, 1) = 42;
    auto sub0  = cuda::std::submdspan(m, 1, cuda::std::full_extent, cuda::std::full_extent);

    static_assert(sub0.rank() == 2, "");
    static_assert(sub0.rank_dynamic() == 0, "");
    assert(sub0.extent(0) == 3);
    assert(sub0.extent(1) == 4);
    assert(sub0(1, 1) == 42);

    auto sub1 = cuda::std::submdspan(sub0, 1, cuda::std::full_extent);
    static_assert(sub1.rank() == 1, "");
    static_assert(sub1.rank_dynamic() == 0, "");
    assert(sub1.extent(0) == 4);
    assert(sub1(1) == 42);

    auto sub2 = cuda::std::submdspan(sub1, 1);
    static_assert(sub2.rank() == 0, "");
    static_assert(sub2.rank_dynamic() == 0, "");
    assert(sub2() == 42);
  }

  return 0;
}
