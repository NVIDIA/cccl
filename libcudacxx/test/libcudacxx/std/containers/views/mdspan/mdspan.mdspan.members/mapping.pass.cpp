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
  // mapping
  {
    using data_t  = int;
    using index_t = size_t;
    cuda::std::array<data_t, 1> d{42};
    cuda::std::layout_left::mapping<cuda::std::extents<index_t, dyn, dyn>> map{
      cuda::std::dextents<index_t, 2>{64, 128}};
    cuda::std::mdspan<data_t, cuda::std::extents<index_t, dyn, dyn>, cuda::std::layout_left> m{d.data(), map};

    assert(m.mapping() == map);
  }

  // mapping and accessor
  {
    using data_t  = int;
    using index_t = size_t;
    cuda::std::array<data_t, 1> d{42};
    cuda::std::layout_left::mapping<cuda::std::extents<index_t, dyn, dyn>> map{
      cuda::std::dextents<index_t, 2>{64, 128}};
    cuda::std::default_accessor<data_t> a;
    cuda::std::mdspan<data_t, cuda::std::extents<index_t, dyn, dyn>, cuda::std::layout_left> m{d.data(), map, a};

    assert(m.mapping() == map);
  }

  return 0;
}
