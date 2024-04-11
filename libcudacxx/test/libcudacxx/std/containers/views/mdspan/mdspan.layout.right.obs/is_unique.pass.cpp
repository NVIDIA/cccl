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

  {
    cuda::std::layout_right::mapping<cuda::std::dextents<index_t, 1>> m;

    static_assert(m.is_always_unique() == true, "");
    assert(m.is_unique() == true);
  }

  {
    cuda::std::extents<index_t, dyn, dyn> e{16, 32};
    cuda::std::layout_right::mapping<cuda::std::extents<index_t, dyn, dyn>> m{e};

    static_assert(m.is_always_unique() == true, "");
    assert(m.is_unique() == true);
  }

  return 0;
}
