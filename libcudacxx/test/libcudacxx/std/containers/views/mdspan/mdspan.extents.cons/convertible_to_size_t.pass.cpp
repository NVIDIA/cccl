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

__host__ __device__ void check(cuda::std::dextents<size_t, 2> e)
{
  static_assert(e.rank() == 2, "");
  static_assert(e.rank_dynamic() == 2, "");

  assert(e.extent(0) == 2);
  assert(e.extent(1) == 2);
}

int main(int, char**)
{
  // TEST(TestExtentsCtorStdArrayConvertibleToSizeT, test_extents_ctor_std_array_convertible_to_size_t)
  {
    cuda::std::array<int, 2> i{2, 2};
    cuda::std::dextents<size_t, 2> e{i};

    check(e);
  }

  // TEST(TestExtentsCtorStdArrayConvertibleToSizeT, test_extents_ctor_std_span_convertible_to_size_t)
  {
    cuda::std::array<int, 2> i{2, 2};
    cuda::std::span<int, 2> s(i.data(), 2);
    cuda::std::dextents<size_t, 2> e{s};

    check(e);
  }

  return 0;
}
