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

#include "../mdspan.layout.util/layout_util.hpp"
#include <test_macros.h>

constexpr auto dyn = cuda::std::dynamic_extent;

int main(int, char**)
{
  using index_t = size_t;
  using ext0d_t = cuda::std::extents<index_t>;
  using ext2d_t = cuda::std::extents<index_t, dyn, dyn>;

  {
    ext2d_t e{64, 128};
    cuda::std::layout_left::mapping<ext2d_t> m{e};

    assert(m.stride(0) == 1);
    assert(m.stride(1) == 64);

    static_assert(is_stride_avail_v<decltype(m), index_t> == true, "");
  }

  {
    ext2d_t e{1, 128};
    cuda::std::layout_left::mapping<ext2d_t> m{e};

    assert(m.stride(0) == 1);
    assert(m.stride(1) == 1);
  }

  // constraint: extents_­type?::?rank() > 0
  {
    ext0d_t e{};
    cuda::std::layout_left::mapping<ext0d_t> m{e};

    unused(m);

    static_assert(is_stride_avail_v<decltype(m), index_t> == false, "");
  }

  return 0;
}
