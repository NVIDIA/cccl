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

int main(int, char**)
{
#ifdef __MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION
  // TEST(TestMdspanCTAD, extents_pack)
  {
    cuda::std::array<int, 1> d{42};
    cuda::std::mdspan m(d.data(), 64, 128);

    static_assert(m.is_exhaustive() == true, "");

    assert(m.data_handle() == d.data());
    assert(m.rank() == 2);
    assert(m.rank_dynamic() == 2);
    assert(m.extent(0) == 64);
    assert(m.extent(1) == 128);
  }
#endif

  return 0;
}
