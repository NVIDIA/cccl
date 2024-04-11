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

  auto e     = cuda::std::dextents<index_t, 2>{16, 32};
  auto s_arr = cuda::std::array<index_t, 2>{1, 128};

  // From a span
  {
    cuda::std::span<index_t, 2> s(s_arr.data(), 2);
    cuda::std::layout_stride::mapping<ext2d_t> m{e, s};

    assert(m.stride(0) == 1);
    assert(m.stride(1) == 128);

    static_assert(is_stride_avail_v<decltype(m), index_t> == true, "");
  }

  // From an array
  {
    cuda::std::layout_stride::mapping<ext2d_t> m{e, s_arr};

    assert(m.stride(0) == 1);
    assert(m.stride(1) == 128);
  }

  // From another mapping
  {
    cuda::std::layout_stride::mapping<ext2d_t> m0{e, s_arr};
    cuda::std::layout_stride::mapping<ext2d_t> m{m0};

    assert(m.stride(0) == 1);
    assert(m.stride(1) == 128);
  }

  // constraint: extents_­type?::?rank() > 0
  {
    ext0d_t e0d{};
    cuda::std::layout_stride::mapping<ext0d_t> m{e0d, cuda::std::array<index_t, 0>{}};

    unused(m);

    static_assert(is_stride_avail_v<decltype(m), index_t> == false, "");
  }

  return 0;
}
