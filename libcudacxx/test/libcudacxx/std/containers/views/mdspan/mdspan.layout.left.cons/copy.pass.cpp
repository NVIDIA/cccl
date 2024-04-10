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

constexpr auto dyn = cuda::std::dynamic_extent;

int main(int, char**)
{
  using index_t = size_t;

  {
    cuda::std::layout_left::mapping<cuda::std::extents<index_t, dyn, dyn>> m0{cuda::std::dextents<index_t, 2>{16, 32}};
    cuda::std::layout_left::mapping<cuda::std::extents<index_t, dyn, dyn>> m{m0};

    static_assert(m.is_exhaustive() == true, "");
    static_assert(m.extents().rank() == 2, "");
    static_assert(m.extents().rank_dynamic() == 2, "");

    assert(m.extents().extent(0) == 16);
    assert(m.extents().extent(1) == 32);
    assert(m.stride(0) == 1);
    assert(m.stride(1) == 16);
  }

  // Constraint: is_constructible_v<extents_type, OtherExtents> is true
  {
    using mapping0_t = cuda::std::layout_left::mapping<cuda::std::extents<index_t, 16, 32>>;
    using mapping1_t = cuda::std::layout_left::mapping<cuda::std::extents<index_t, 16, 16>>;
    using mappingd_t = cuda::std::layout_left::mapping<cuda::std::dextents<index_t, 2>>;

    static_assert(is_cons_avail_v<mappingd_t, mapping0_t> == true, "");
    static_assert(is_cons_avail_v<mapping1_t, mapping0_t> == false, "");
  }

  return 0;
}
