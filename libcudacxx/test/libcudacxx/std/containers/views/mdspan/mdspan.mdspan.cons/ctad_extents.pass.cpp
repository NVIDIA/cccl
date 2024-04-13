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

#define CHECK_MDSPAN(m, d)              \
  static_assert(m.is_exhaustive(), ""); \
  assert(m.data_handle() == d.data());  \
  assert(m.rank() == 2);                \
  assert(m.rank_dynamic() == 2);        \
  assert(m.extent(0) == 64);            \
  assert(m.extent(1) == 128)

int main(int, char**)
{
#ifdef __MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION
  // TEST(TestMdspanCTAD, extents_object)
  {
    cuda::std::array<int, 1> d{42};
    cuda::std::mdspan m{d.data(), cuda::std::extents{64, 128}};

    CHECK_MDSPAN(m, d);
  }

  // TEST(TestMdspanCTAD, extents_object_move)
  {
    cuda::std::array<int, 1> d{42};
    cuda::std::mdspan m{d.data(), std::move(cuda::std::extents{64, 128})};

    CHECK_MDSPAN(m, d);
  }

  // TEST(TestMdspanCTAD, extents_std_array)
  {
    cuda::std::array<int, 1> d{42};
    cuda::std::mdspan m{d.data(), cuda::std::array{64, 128}};

    CHECK_MDSPAN(m, d);
  }

  // TEST(TestMdspanCTAD, cptr_extents_std_array)
  {
    cuda::std::array<int, 1> d{42};
    const int* const ptr = d.data();
    cuda::std::mdspan m{ptr, cuda::std::array{64, 128}};

    static_assert(cuda::std::is_same<typename decltype(m)::element_type, const int>::value, "");

    CHECK_MDSPAN(m, d);
  }

  // extents from std::span
  {
    cuda::std::array<int, 1> d{42};
    cuda::std::array<int, 2> sarr{64, 128};
    cuda::std::mdspan m{d.data(), cuda::std::span{sarr}};

    CHECK_MDSPAN(m, d);
  }
#endif

  return 0;
}
