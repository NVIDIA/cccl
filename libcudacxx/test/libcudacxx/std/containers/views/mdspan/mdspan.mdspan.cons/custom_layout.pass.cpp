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

#include "../foo_customizations.hpp"
#include "../mdspan.mdspan.util/mdspan_util.hpp"

constexpr auto dyn = cuda::std::dynamic_extent;

int main(int, char**)
{
  {
    using data_t  = int;
    using lay_t   = Foo::layout_foo;
    using index_t = size_t;

    cuda::std::array<data_t, 1> d{42};
    lay_t::mapping<cuda::std::extents<index_t, dyn, dyn>> map{cuda::std::dextents<index_t, 2>{64, 128}};
    cuda::std::mdspan<data_t, cuda::std::extents<index_t, dyn, dyn>, lay_t> m{d.data(), map};

    CHECK_MDSPAN_EXTENT(m, d, 64, 128);
  }

  return 0;
}
