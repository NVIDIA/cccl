//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// template<> class complex<float>
// {
// public:
//     explicit constexpr complex(const complex<double>&);
// };

#if defined(__clang__)
#  define _CCCL_IMPLICIT_SYSTEM_HEADER_CLANG
#elif defined(_MSC_VER)
#  define _CCCL_IMPLICIT_SYSTEM_HEADER_MSVC
#else
#  define _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#endif

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "test_macros.h"

int main(int, char**)
{
  {
    const cuda::std::complex<double> cd(2.5, 3.5);
    cuda::std::complex<float> cf(cd);
    assert(cf.real() == cd.real());
    assert(cf.imag() == cd.imag());
  }
  {
    constexpr cuda::std::complex<double> cd(2.5, 3.5);
    constexpr cuda::std::complex<float> cf(cd);
    static_assert(cf.real() == cd.real(), "");
    static_assert(cf.imag() == cd.imag(), "");
  }

  static_assert(cuda::std::is_same<cuda::std::common_type<cuda::std::complex<float>, cuda::std::complex<double>>::type,
                                   cuda::std::complex<cuda::std::common_type<float, double>::type>>::value,
                "");

  return 0;
}
