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

    static_assert(m.rank() == 1, "");
    assert(m.rank_dynamic() == 1);
  }

  {
    cuda::std::mdspan<data_t, cuda::std::extents<index_t, 16>> m{d.data()};

    static_assert(m.rank() == 1, "");
    assert(m.rank_dynamic() == 0);
  }

  {
    cuda::std::mdspan<data_t, cuda::std::extents<index_t, dyn, dyn>> m{d.data(), 16, 32};

    static_assert(m.rank() == 2, "");
    assert(m.rank_dynamic() == 2);
  }

  {
    cuda::std::mdspan<data_t, cuda::std::extents<index_t, 8, dyn, dyn>> m{d.data(), 16, 32};

    static_assert(m.rank() == 3, "");
    assert(m.rank_dynamic() == 2);
  }

  return 0;
}
