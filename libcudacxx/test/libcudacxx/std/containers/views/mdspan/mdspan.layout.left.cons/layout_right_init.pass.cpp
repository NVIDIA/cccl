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
    cuda::std::layout_right::mapping<cuda::std::extents<index_t, dyn>> m_right{cuda::std::dextents<index_t, 1>{16}};
    cuda::std::layout_left ::mapping<cuda::std::extents<index_t, dyn>> m(m_right);

    static_assert(m.is_exhaustive() == true, "");

    assert(m.extents().rank() == 1);
    assert(m.extents().rank_dynamic() == 1);
    assert(m.extents().extent(0) == 16);
    assert(m.stride(0) == 1);
  }

  // Constraint: extents_type::rank() <= 1 is true
  {
    using mapping0_t = cuda::std::layout_right::mapping<cuda::std::extents<index_t, 16, 32>>;
    using mapping1_t = cuda::std::layout_left ::mapping<cuda::std::extents<index_t, 16, 32>>;

    static_assert(is_cons_avail_v<mapping1_t, mapping0_t> == false, "");
  }

  // Constraint: is_constructible_v<extents_type, OtherExtents> is true
  {
    using mapping0_t = cuda::std::layout_right::mapping<cuda::std::extents<index_t, 16>>;
    using mapping1_t = cuda::std::layout_left ::mapping<cuda::std::extents<index_t, 32>>;
    using mappingd_t = cuda::std::layout_left ::mapping<cuda::std::dextents<index_t, 1>>;

    static_assert(is_cons_avail_v<mappingd_t, mapping0_t> == true, "");
    static_assert(is_cons_avail_v<mapping1_t, mapping0_t> == false, "");
  }

  return 0;
}
