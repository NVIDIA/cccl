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
  using index_t = int;
  using ext2d_t = cuda::std::extents<index_t, dyn, dyn>;

  {
    cuda::std::extents<index_t, 16, 32> e;
    cuda::std::array<index_t, 2> a{1, 16};
    cuda::std::layout_stride::mapping<ext2d_t> m0{e, a};
    cuda::std::layout_stride::mapping<ext2d_t> m{m0};

    assert(m0 == m);
  }

  {
    using index2_t = int32_t;

    cuda::std::extents<index_t, 16, 32> e;
    cuda::std::array<index_t, 2> a{1, 16};
    cuda::std::extents<index2_t, 16, 32> e2;
    cuda::std::array<index2_t, 2> a2{1, 16};
    cuda::std::layout_stride::mapping<ext2d_t> m1{e, a};
    cuda::std::layout_stride::mapping<ext2d_t> m2{e2, a2};

    assert(m1 == m2);
  }

  {
    cuda::std::extents<index_t, 16, 32> e;
    cuda::std::array<index_t, 2> a0{1, 16};
    cuda::std::array<index_t, 2> a1{1, 32};
    cuda::std::layout_stride::mapping<ext2d_t> m0{e, a0};
    cuda::std::layout_stride::mapping<ext2d_t> m1{e, a1};

    assert(m0 != m1);
  }

  {
    cuda::std::extents<index_t, 16, 32> e;
    cuda::std::array<index_t, 2> a{1, 16};
    cuda::std::layout_stride::mapping<ext2d_t> m{e, a};
    cuda::std::layout_left ::mapping<ext2d_t> m_left{e};

    assert(m == m_left);
  }

  {
    cuda::std::extents<index_t, 16, 32> e;
    cuda::std::array<index_t, 2> a{32, 1};
    cuda::std::layout_stride::mapping<ext2d_t> m{e, a};
    cuda::std::layout_right ::mapping<ext2d_t> m_right{e};

    assert(m == m_right);
  }

  {
    cuda::std::extents<index_t, 16, 32> e0;
    cuda::std::extents<index_t, 16, 64> e1;
    cuda::std::array<index_t, 2> a{1, 16};
    cuda::std::layout_stride::mapping<ext2d_t> m0{e0, a};
    cuda::std::layout_stride::mapping<ext2d_t> m1{e1, a};

    assert(m0 != m1);
  }

  return 0;
}
