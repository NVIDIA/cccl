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

#include <cuda/std/mdspan>

int main(int, char**)
{
  using index_t = size_t;
  using ext2d_t = cuda::std::extents<index_t, 64, 128>;
  using ext3d_t = cuda::std::extents<index_t, 64, 128, 2>;

  // Constraint: rank consistency
  // This constraint is implemented in a different way in the reference implementation. There will be an overload
  // function match but it will return false if the ranks are not consistent
  {
    constexpr ext2d_t e0;
    constexpr ext3d_t e1;
    constexpr cuda::std::layout_left::mapping<ext2d_t> m0{e0};
    constexpr cuda::std::layout_left::mapping<ext3d_t> m1{e1};

    static_assert(m0 == m1, ""); // expected-error
  }

  return 0;
}
