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

int main(int, char**)
{
  // C array
  {
    const int d[5] = {1, 2, 3, 4, 5};
#if defined(__cpp_deduction_guides) && defined(__MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
    cuda::std::mdspan m(d);
#else
    cuda::std::mdspan<const int, cuda::std::extents<size_t, 5>> m(d);
#endif

    assert(m.data_handle() == d);
  }

  // std array
  {
    cuda::std::array<int, 5> d = {1, 2, 3, 4, 5};
#if defined(__cpp_deduction_guides) && defined(__MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
    cuda::std::mdspan m(d.data());
#else
    cuda::std::mdspan<int, cuda::std::extents<size_t, 5>> m(d.data());
#endif

    assert(m.data_handle() == d.data());
  }

  // C pointer
  {
    cuda::std::array<int, 5> d = {1, 2, 3, 4, 5};
    int* ptr                   = d.data();
#if defined(__cpp_deduction_guides) && defined(__MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
    cuda::std::mdspan m(ptr);
#else
    cuda::std::mdspan<int, cuda::std::extents<size_t, 5>> m(ptr);
#endif

    assert(m.data_handle() == ptr);
  }

  // Copy constructor
  {
    cuda::std::array<int, 5> d = {1, 2, 3, 4, 5};
#if defined(__cpp_deduction_guides) && defined(__MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
    cuda::std::mdspan m0(d.data());
    cuda::std::mdspan m(m0);
#else
    cuda::std::mdspan<int, cuda::std::extents<size_t, 5>> m0(d.data());
    cuda::std::mdspan<int, cuda::std::extents<size_t, 5>> m(m0);
#endif

    assert(m.data_handle() == m0.data_handle());
  }

  return 0;
}
