//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc && c++14, msvc && c++17

#include <cuda/std/cassert>
#include <cuda/std/mdspan>

constexpr auto dyn = cuda::std::dynamic_extent;

int main(int, char**)
{
  using index_t = int;

  {
    cuda::std::extents<index_t, 16> e;
    cuda::std::array<index_t, 1> a{1};
    cuda::std::layout_stride::mapping<cuda::std::extents<index_t, 16>> m{e, a};

    static_assert(m.is_always_exhaustive() == false, "");
    assert(m.is_exhaustive() == true);
  }

  {
    cuda::std::extents<index_t, 16> e;
    cuda::std::array<index_t, 1> a{2};
    cuda::std::layout_stride::mapping<cuda::std::extents<index_t, 16>> m{e, a};

    static_assert(m.is_always_exhaustive() == false, "");
    assert(m.is_exhaustive() == false);
  }

  {
    cuda::std::extents<index_t, 16, 32> e;
    cuda::std::array<index_t, 2> a{1, 16};
    cuda::std::layout_stride::mapping<cuda::std::extents<index_t, dyn, dyn>> m{e, a};

    static_assert(m.is_always_exhaustive() == false, "");
    assert(m.is_exhaustive() == true);
  }

  {
    cuda::std::extents<index_t, dyn, dyn> e{16, 32};
    cuda::std::array<index_t, 2> a{1, 128};
    cuda::std::layout_stride::mapping<cuda::std::extents<index_t, dyn, dyn>> m{e, a};

    static_assert(m.is_always_exhaustive() == false, "");
    assert(m.is_exhaustive() == false);
  }

  {
    cuda::std::extents<index_t, dyn, dyn, dyn> e{16, 32, 4};
    cuda::std::array<index_t, 3> a{1, 16 * 4, 16};
    cuda::std::layout_stride::mapping<cuda::std::extents<index_t, dyn, dyn, dyn>> m{e, a};

    static_assert(m.is_always_exhaustive() == false, "");
    assert(m.is_exhaustive() == true);
  }

  {
    cuda::std::extents<index_t, dyn, dyn, dyn> e{16, 32, 4};
    cuda::std::array<index_t, 3> a{1, 16 * 4 + 1, 16};
    cuda::std::layout_stride::mapping<cuda::std::extents<index_t, dyn, dyn, dyn>> m{e, a};

    static_assert(m.is_always_exhaustive() == false, "");
    assert(m.is_exhaustive() == false);
  }

  return 0;
}
