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

#include <cuda/std/cassert>
#include <cuda/std/complex>

int main(int, char**)
{
  const cuda::std::complex<double> cd(2.5, 3.5);
  cuda::std::complex<float> cf = cd;
  assert(cf.real() == cd.real());
  assert(cf.imag() == cd.imag());

  return 0;
}
