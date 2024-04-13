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
  // TEST(TestSubmdspanLayoutRightStaticSizedPairs, test_submdspan_layout_right_static_sized_pairs)
  {
    cuda::std::array<int, 2 * 3 * 4> d;
    cuda::std::mdspan<int, cuda::std::extents<size_t, 2, 3, 4>> m(d.data());
    m(1, 1, 1) = 42;
    auto sub0  = cuda::std::submdspan(
      m, cuda::std::pair<int, int>{1, 2}, cuda::std::pair<int, int>{1, 3}, cuda::std::pair<int, int>{1, 4});

    static_assert(sub0.rank() == 3, "");
    static_assert(sub0.rank_dynamic() == 3, "");
    assert(sub0.extent(0) == 1);
    assert(sub0.extent(1) == 2);
    assert(sub0.extent(2) == 3);
    assert(sub0(0, 0, 0) == 42);
  }

  return 0;
}
