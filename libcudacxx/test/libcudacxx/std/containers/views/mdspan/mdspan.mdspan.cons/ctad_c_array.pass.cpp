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
  // TEST(TestMdspanCTAD, ctad_carray)
  {
    int data[5] = {1, 2, 3, 4, 5};
    cuda::std::mdspan m(data);

    static_assert(cuda::std::is_same<decltype(m)::element_type, int>::value == true, "");
    static_assert(m.is_exhaustive() == true, "");

    assert(m.data_handle() == &data[0]);
    assert(m.rank() == 1);
    assert(m.rank_dynamic() == 0);
    assert(m.static_extent(0) == 5);
    assert(m.extent(0) == 5);
    assert(__MDSPAN_OP(m, 2) == 3);

    cuda::std::mdspan m2(data, 3);

    static_assert(cuda::std::is_same<decltype(m2)::element_type, int>::value == true, "");
    static_assert(m2.is_exhaustive() == true, "");

    assert(m2.data_handle() == &data[0]);
    assert(m2.rank() == 1);
    assert(m2.rank_dynamic() == 1);
    assert(m2.extent(0) == 3);
    assert(__MDSPAN_OP(m2, 2) == 3);
  }
#endif

  return 0;
}
