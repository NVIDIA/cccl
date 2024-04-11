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
#include "../my_int.hpp"

constexpr auto dyn = cuda::std::dynamic_extent;

int main(int, char**)
{
  using index_t = int;

  {
    cuda::std::extents<index_t, 16> e;
    cuda::std::array<index_t, 1> a{1};
    cuda::std::layout_stride::mapping<cuda::std::extents<index_t, 16>> m{e, a};

    assert(m(8) == 8);
  }

  {
    cuda::std::extents<index_t, dyn, dyn> e{16, 32};
    cuda::std::array<index_t, 2> a{1, 16};
    cuda::std::layout_stride::mapping<cuda::std::extents<index_t, dyn, dyn>> m{e, a};

    assert(m(8, 16) == 8 * 1 + 16 * 16);
  }

  {
    cuda::std::extents<index_t, 16, dyn> e{32};
    cuda::std::array<index_t, 2> a{1, 24};
    cuda::std::layout_stride::mapping<cuda::std::extents<index_t, dyn, dyn>> m{e, a};

    assert(m(8, 16) == 8 * 1 + 16 * 24);
  }

  {
    cuda::std::extents<index_t, 16, dyn> e{32};
    cuda::std::array<index_t, 2> a{48, 1};
    cuda::std::layout_stride::mapping<cuda::std::extents<index_t, dyn, dyn>> m{e, a};

    assert(m(8, 16) == 8 * 48 + 16 * 1);
  }

  // Indices are of a type implicitly convertible to index_type
  {
    cuda::std::extents<index_t, dyn, dyn> e{16, 32};
    cuda::std::array<index_t, 2> a{1, 16};
    cuda::std::layout_stride::mapping<cuda::std::extents<index_t, dyn, dyn>> m{e, a};

    assert(m(my_int(8), my_int(16)) == 8 * 1 + 16 * 16);
  }

  // Constraints
  {
    cuda::std::extents<index_t, 16> e;
    cuda::std::array<index_t, 1> a{1};
    cuda::std::layout_stride::mapping<cuda::std::extents<index_t, 16>> m{e, a};

    static_assert(is_paren_op_avail_v<decltype(m), index_t> == true, "");

    // rank consistency
    static_assert(is_paren_op_avail_v<decltype(m), index_t, index_t> == false, "");

    // convertibility
    static_assert(is_paren_op_avail_v<decltype(m), my_int_non_convertible> == false, "");

    // nothrow-constructibility
#ifndef TEST_COMPILER_BROKEN_SMF_NOEXCEPT
    static_assert(is_paren_op_avail_v<decltype(m), my_int_non_nothrow_constructible> == false, "");
#endif // TEST_COMPILER_BROKEN_SMF_NOEXCEPT
  }

  return 0;
}
