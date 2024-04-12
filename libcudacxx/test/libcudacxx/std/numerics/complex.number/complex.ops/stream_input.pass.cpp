//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-localization
// UNSUPPORTED: nvrtc

// <complex>

// template<class T, class charT, class traits>
//   basic_istream<charT, traits>&
//   operator>>(basic_istream<charT, traits>& is, complex<T>& x);

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include <sstream>

#include "test_macros.h"

template <class T>
void test()
{
  {
    std::istringstream is("5");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(5, 0));
    assert(is.eof());
  }
  {
    std::istringstream is(" 5 ");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(5, 0));
    assert(is.good());
  }
  {
    std::istringstream is(" 5, ");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(5, 0));
    assert(is.good());
  }
  {
    std::istringstream is(" , 5, ");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(0, 0));
    assert(is.fail());
  }
  {
    std::istringstream is("5.5 ");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(5.5, 0));
    assert(is.good());
  }
  {
    std::istringstream is(" ( 5.5 ) ");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(5.5, 0));
    assert(is.good());
  }
  {
    std::istringstream is("  5.5)");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(5.5, 0));
    assert(is.good());
  }
  {
    std::istringstream is("(5.5 ");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(0, 0));
    assert(is.fail());
  }
  {
    std::istringstream is("(5.5,");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(0, 0));
    assert(is.fail());
  }
  {
    std::istringstream is("( -5.5 , -6.5 )");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(-5.5, -6.5));
    assert(!is.eof());
  }
  {
    std::istringstream is("(-5.5,-6.5)");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(-5.5, -6.5));
    assert(!is.eof());
  }
}

void test()
{
  test<float>();
  test<double>();
#ifdef _LIBCUDACXX_HAS_NVFP16
  test<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16
#ifdef _LIBCUDACXX_HAS_NVBF16
  test<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, test();)
  return 0;
}
