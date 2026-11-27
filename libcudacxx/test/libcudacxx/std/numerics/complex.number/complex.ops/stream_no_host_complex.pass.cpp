//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-localization
// UNSUPPORTED: nvrtc

// <cuda/std/complex>

// Streaming a cuda::std::complex must work without the host <complex> being included

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include <sstream>

#include "test_macros.h"

template <class T>
void test()
{
  {
    std::ostringstream os;
    os << cuda::std::complex<T>(1, 2);
    assert(os.str() == "(1,2)");
  }
  {
    std::istringstream is("(3,4)");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(3, 4));
  }
  {
    std::stringstream ss;
    ss << cuda::std::complex<T>(5, 6);
    cuda::std::complex<T> c;
    ss >> c;
    assert(c == cuda::std::complex<T>(5, 6));
  }
}

void test()
{
  test<float>();
  test<double>();
#if _LIBCUDACXX_HAS_NVFP16()
  test<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, test();)
  return 0;
}
