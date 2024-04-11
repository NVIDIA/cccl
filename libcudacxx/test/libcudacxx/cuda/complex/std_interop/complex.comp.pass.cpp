//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include <complex>

#include "test_macros.h"
#include <nv/target>

template <class T, class U>
void test_comparison()
{
  ::cuda::std::complex<T> input{static_cast<T>(-1.0), static_cast<T>(1.0)};

  const ::std::complex<U> not_equal_real{static_cast<T>(-1.0), 0};
  const ::std::complex<U> not_equal_imag{0, static_cast<T>(1.0)};
  const ::std::complex<U> equal{static_cast<T>(-1.0), static_cast<T>(1.0)};

  assert(!(input == not_equal_real));
  assert(!(input == not_equal_imag));
  assert(input == equal);

  assert(input != not_equal_real);
  assert(input != not_equal_imag);
  assert(!(input != equal));
}

void test()
{
  test_comparison<float, float>();
  test_comparison<double, float>();
  test_comparison<double, double>();
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();));

  return 0;
}
