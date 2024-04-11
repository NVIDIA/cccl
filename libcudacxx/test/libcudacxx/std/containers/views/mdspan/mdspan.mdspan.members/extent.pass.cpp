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
  typedef int data_t;
  typedef size_t index_t;

  cuda::std::array<data_t, 1> d{42};

  {
    cuda::std::mdspan<data_t, cuda::std::dextents<index_t, 1>> m;

    assert(m.extent(0) == 0);
    assert(m.extent(1) == 0);
  }

  {
    cuda::std::mdspan<data_t, cuda::std::extents<index_t, 16, 32>> m{d.data()};

    assert(m.extent(0) == 16);
    assert(m.extent(1) == 32);
  }

  {
    cuda::std::mdspan<data_t, cuda::std::extents<index_t, dyn, dyn>> m{d.data(), 16, 32};

    assert(m.extent(0) == 16);
    assert(m.extent(1) == 32);
  }

  {
    cuda::std::mdspan<data_t, cuda::std::extents<index_t, dyn, dyn>, cuda::std::layout_left> m{d.data(), 16, 32};

    assert(m.extent(0) == 16);
    assert(m.extent(1) == 32);
  }

  {
    using dexts = cuda::std::dextents<size_t, 2>;

    cuda::std::mdspan<int, cuda::std::extents<size_t, dyn, dyn>, cuda::std::layout_stride> m{
      d.data(), cuda::std::layout_stride::template mapping<dexts>{dexts{16, 32}, cuda::std::array<size_t, 2>{1, 128}}};

    assert(m.extent(0) == 16);
    assert(m.extent(1) == 32);
  }

  return 0;
}
