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
  using ext1d_t = cuda::std::extents<index_t, dyn>;
  using ext2d_t = cuda::std::extents<index_t, dyn, dyn>;

  {
    ext2d_t e{16, 32};
    cuda::std::layout_left::mapping<ext2d_t> m{e};

    assert(m.extents() == e);
  }

  {
    ext1d_t e{16};
    cuda::std::layout_right::mapping<ext1d_t> m_right{e};
    cuda::std::layout_left ::mapping<ext1d_t> m{m_right};

    assert(m.extents() == e);
  }

  {
    ext2d_t e{16, 32};
    cuda::std::array<index_t, 2> a{1, 16};
    cuda::std::layout_stride::mapping<ext2d_t> m_stride{e, a};
    cuda::std::layout_left ::mapping<ext2d_t> m{m_stride};

    assert(m.extents() == e);
  }

  return 0;
}
