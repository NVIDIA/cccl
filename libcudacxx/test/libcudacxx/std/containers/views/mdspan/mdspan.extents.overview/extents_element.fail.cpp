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

__host__ __device__ void check(cuda::std::dextents<size_t, 2> e)
{
  static_assert(e.rank() == 2, "");
  static_assert(e.rank_dynamic() == 2, "");

  assert(e.extent(0) == 2);
  assert(e.extent(1) == 2);
}

struct dummy
{};

int main(int, char**)
{
  {
    cuda::std::dextents<int, 2> e{2, 2};

    check(e);
  }

  // Mandate: each element of Extents is either equal to dynamic_extent, or is representable as a value of type
  // IndexType
  {
    cuda::std::dextents<int, 2> e{dummy{}, 2};

    check(e);
  }

  return 0;
}
