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

#include "../mdspan.layout.util/layout_util.hpp"

constexpr auto dyn = cuda::std::dynamic_extent;

__host__ __device__ void typed_test_compare_right()
{
  typed_test_compare<test_right_type_pair<_exts<dyn>, _sizes<10>, _exts<10>, _sizes<>>>();
  typed_test_compare<test_right_type_pair<_exts<dyn, 10>, _sizes<5>, _exts<5, dyn>, _sizes<10>>>();
  typed_test_compare<test_right_type_pair<_exts<dyn, dyn>, _sizes<5, 10>, _exts<5, dyn>, _sizes<10>>>();
  typed_test_compare<test_right_type_pair<_exts<dyn, dyn>, _sizes<5, 10>, _exts<dyn, 10>, _sizes<5>>>();
  typed_test_compare<test_right_type_pair<_exts<dyn, dyn>, _sizes<5, 10>, _exts<5, 10>, _sizes<>>>();
  typed_test_compare<test_right_type_pair<_exts<5, 10>, _sizes<>, _exts<5, dyn>, _sizes<10>>>();
  typed_test_compare<test_right_type_pair<_exts<5, 10>, _sizes<>, _exts<dyn, 10>, _sizes<5>>>();
  typed_test_compare<test_right_type_pair<_exts<dyn, dyn, 15>, _sizes<5, 10>, _exts<5, dyn, 15>, _sizes<10>>>();
  typed_test_compare<test_right_type_pair<_exts<5, 10, 15>, _sizes<>, _exts<5, dyn, 15>, _sizes<10>>>();
  typed_test_compare<test_right_type_pair<_exts<5, 10, 15>, _sizes<>, _exts<dyn, dyn, dyn>, _sizes<5, 10, 15>>>();
}

int main(int, char**)
{
  typed_test_compare_right();

  using index_t = size_t;
  using ext2d_t = cuda::std::extents<index_t, dyn, dyn>;

  {
    ext2d_t e{64, 128};
    cuda::std::layout_right::mapping<ext2d_t> m0{e};
    cuda::std::layout_right::mapping<ext2d_t> m{m0};

    assert(m == m0);
  }

  {
    ext2d_t e0{64, 128};
    ext2d_t e1{16, 32};
    cuda::std::layout_right::mapping<ext2d_t> m0{e0};
    cuda::std::layout_right::mapping<ext2d_t> m1{e1};

    assert(m0 != m1);
  }

  return 0;
}
