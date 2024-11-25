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

#include <cuda/std/__linalg/scaled.h>
#include <cuda/std/cassert>

int main(int, char**)
{
  {
    using element_t = int;
    cuda::std::array<element_t, 2> d{42, 43};
    cuda::std::mdspan md(d.data(), 2);
    auto scaled_md = cuda::std::linalg::scaled(2, md); // scaled

    assert(scaled_md(0) == 42 * 2);
    assert(scaled_md(1) == 43 * 2);
  }

  return 0;
}
