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

constexpr auto dyn = cuda::std::dynamic_extent;

int main(int, char**)
{
  using index_t = size_t;
  using ext2d_t = cuda::std::extents<index_t, 64, 128>;
  using ext3d_t = cuda::std::extents<index_t, 64, 128, 2>;

  // Constraint: rank consistency
  {
    constexpr ext2d_t e0;
    constexpr ext3d_t e1;
    constexpr cuda::std::array<index_t, 2> a0{1, 64};
    constexpr cuda::std::array<index_t, 3> a1{1, 64, 64 * 128};
    constexpr cuda::std::layout_stride::mapping<ext2d_t> m0{e0, a0};
    constexpr cuda::std::layout_stride::mapping<ext3d_t> m1{e1, a1};

    static_assert(m0 == m1, ""); // expected-error
  }

  return 0;
}
