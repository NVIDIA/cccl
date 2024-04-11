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
  cuda::std::array<int, 1> d{42};

  cuda::std::mdspan<int, cuda::std::extents<size_t, dyn, dyn>, cuda::std::layout_stride> m{
    d.data(),
    cuda::std::layout_stride::template mapping<cuda::std::dextents<size_t, 2>>{
      cuda::std::dextents<size_t, 2>{16, 32}, cuda::std::array<size_t, 2>{1, 128}}};

  assert(m.data_handle() == d.data());
  assert(m.rank() == 2);
  assert(m.rank_dynamic() == 2);
  assert(m.extent(0) == 16);
  assert(m.extent(1) == 32);
  assert(m.stride(0) == 1);
  assert(m.stride(1) == 128);
  assert(m.is_exhaustive() == false);

  return 0;
}
