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

  // copy constructor
  {
    cuda::std::array<int, 1> d{42};
    cuda::std::mdspan<int, cuda::std::extents<size_t, dyn, dyn>> m0{d.data(), cuda::std::extents{64, 128}};
    cuda::std::mdspan m{m0};

    CHECK_MDSPAN_EXTENT(m, d, 64, 128);
  }
#endif

  return 0;
}
