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
    using index_t = size_t;

    cuda::std::extents<index_t, 10> e0;
    cuda::std::extents<index_t, 10> e1;

    assert(e0 == e1);
  }

  {
    using index_t = size_t;

    cuda::std::extents<index_t, 10> e0;
    cuda::std::extents<index_t, dyn> e1{10};

    assert(e0 == e1);
  }

  {
    using index_t = size_t;

    cuda::std::extents<index_t, 10> e0;
    cuda::std::extents<index_t, 10, 10> e1;

    assert(e0 != e1);
  }

  {
    using index0_t = size_t;
    using index1_t = uint8_t;

    cuda::std::extents<index0_t, 10> e0;
    cuda::std::extents<index1_t, 10> e1;

    assert(e0 == e1);
  }

  return 0;
}
