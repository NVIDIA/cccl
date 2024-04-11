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

    static_assert(m.is_always_unique() == true, "");
    assert(m.is_unique() == true);
  }

  cuda::std::array<int, 1> d{42};
  cuda::std::extents<int, dyn, dyn> e{64, 128};

  {
    cuda::std::mdspan<int, cuda::std::extents<int, dyn, dyn>> m{d.data(), e};

    static_assert(m.is_always_unique() == true, "");
    assert(m.is_unique() == true);
  }

  {
    cuda::std::mdspan<int, cuda::std::extents<size_t, dyn, dyn>, cuda::std::layout_left> m{d.data(), e};

    static_assert(m.is_always_unique() == true, "");
    assert(m.is_unique() == true);
  }

  return 0;
}
