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
  {
    cuda::std::mdspan<int, cuda::std::dextents<size_t, 1>> m;

    static_assert(m.is_always_strided() == true, "");
    assert(m.is_strided() == true);
  }

  cuda::std::array<int, 1> d{42};
  cuda::std::extents<int, dyn, dyn> e{64, 128};

  {
    cuda::std::mdspan<int, cuda::std::extents<size_t, dyn, dyn>, cuda::std::layout_left> m{d.data(), e};

    static_assert(m.is_always_strided() == true, "");
    assert(m.is_strided() == true);
  }

  {
    using dexts = cuda::std::dextents<size_t, 2>;

    cuda::std::mdspan<int, cuda::std::extents<size_t, dyn, dyn>, cuda::std::layout_stride> m{
      d.data(), cuda::std::layout_stride::template mapping<dexts>{dexts{16, 32}, cuda::std::array<size_t, 2>{1, 128}}};

    static_assert(m.is_always_strided() == true, "");
    assert(m.is_strided() == true);
  }

  return 0;
}
