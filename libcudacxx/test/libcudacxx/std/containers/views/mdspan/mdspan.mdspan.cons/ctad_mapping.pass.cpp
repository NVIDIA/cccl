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

// No CTAD in C++14 or earlier
// UNSUPPORTED: c++14

#include <cuda/std/cassert>
#include <cuda/std/mdspan>

#include "../mdspan.mdspan.util/mdspan_util.hpp"

int main(int, char**)
{
#ifdef __MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION
  constexpr auto dyn = cuda::std::dynamic_extent;

  // mapping
  {
    using data_t  = int;
    using index_t = size_t;
    cuda::std::array<data_t, 1> d{42};
    cuda::std::layout_left::mapping<cuda::std::extents<index_t, dyn, dyn>> map{
      cuda::std::dextents<index_t, 2>{64, 128}};
    cuda::std::mdspan m{d.data(), map};

    CHECK_MDSPAN_EXTENT(m, d, 64, 128);
  }

  // mapping and accessor
  {
    using data_t  = int;
    using index_t = size_t;
    cuda::std::array<data_t, 1> d{42};
    cuda::std::layout_left::mapping<cuda::std::extents<index_t, dyn, dyn>> map{
      cuda::std::dextents<index_t, 2>{64, 128}};
    cuda::std::default_accessor<data_t> a;
    cuda::std::mdspan m{d.data(), map, a};

    CHECK_MDSPAN_EXTENT(m, d, 64, 128);
  }
#endif

  return 0;
}
