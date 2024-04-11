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

#define CHECK_MDSPAN(m, d, exhaust, s0, s1) \
  static_assert(m.rank() == 2, "");         \
  static_assert(m.rank_dynamic() == 2, ""); \
  assert(m.data_handle() == d.data());      \
  assert(m.extent(0) == 16);                \
  assert(m.extent(1) == 32);                \
  assert(m.stride(0) == s0);                \
  assert(m.stride(1) == s1);                \
  assert(m.is_exhaustive() == exhaust)

int main(int, char**)
{
#ifdef __MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION
  // TEST(TestMdspanCTAD, layout_left)
  {
    cuda::std::array<int, 1> d{42};
    cuda::std::mdspan m0{d.data(), cuda::std::layout_left::mapping{cuda::std::extents{16, 32}}};

    CHECK_MDSPAN(m0, d, true, 1, 16);
  }

  // TEST(TestMdspanCTAD, layout_right)
  {
    cuda::std::array<int, 1> d{42};
    cuda::std::mdspan m0{d.data(), cuda::std::layout_right::mapping{cuda::std::extents{16, 32}}};

    CHECK_MDSPAN(m0, d, true, 32, 1);
  }

  // TEST(TestMdspanCTAD, layout_stride)
  {
    cuda::std::array<int, 1> d{42};
    cuda::std::mdspan m0{d.data(),
                         cuda::std::layout_stride::mapping{cuda::std::extents{16, 32}, cuda::std::array{1, 128}}};

    CHECK_MDSPAN(m0, d, false, 1, 128);
  }
#endif

  return 0;
}
