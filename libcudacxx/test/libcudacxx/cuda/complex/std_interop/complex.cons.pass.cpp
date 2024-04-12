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
void test_construction()
{
  const ::std::complex<U> only_real{static_cast<U>(42.0), 0};
  const ::std::complex<U> only_imag{0, static_cast<U>(42.0)};
  const ::std::complex<U> real_imag{static_cast<U>(42.0), static_cast<U>(1337.0)};

  const ::cuda::std::complex<T> from_only_real{only_real};
  const ::cuda::std::complex<T> from_only_imag{only_imag};
  const ::cuda::std::complex<T> from_real_imag{real_imag};

  assert(from_only_real.real() == static_cast<T>(42.0));
  assert(from_only_real.imag() == 0);
  assert(from_only_imag.real() == 0);
  assert(from_only_imag.imag() == static_cast<T>(42.0));
  assert(from_real_imag.real() == static_cast<T>(42.0));
  assert(from_real_imag.imag() == static_cast<T>(1337.0));
}

void test()
{
  test_construction<float, float>();
  test_construction<double, float>();
  test_construction<double, double>();
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();));

  return 0;
}
