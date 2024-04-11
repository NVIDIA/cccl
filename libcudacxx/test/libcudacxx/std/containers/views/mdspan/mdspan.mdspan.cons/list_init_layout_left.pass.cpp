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

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/mdspan>

constexpr auto dyn = cuda::std::dynamic_extent;

int main(int, char**)
{
  {
    typedef int data_t;
    typedef size_t index_t;

    cuda::std::array<data_t, 1> d{42};
    cuda::std::mdspan<data_t, cuda::std::extents<index_t, dyn, dyn>, cuda::std::layout_left> m{d.data(), 16, 32};

    static_assert(m.is_exhaustive() == true, "");

    assert(m.data_handle() == d.data());
    assert(m.rank() == 2);
    assert(m.rank_dynamic() == 2);
    assert(m.extent(0) == 16);
    assert(m.extent(1) == 32);
    assert(m.stride(0) == 1);
    assert(m.stride(1) == 16);
  }

  return 0;
}
