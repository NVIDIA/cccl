//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// <numeric>

// template<class _M, class _N>
// constexpr common_type_t<_M,_N> lcm(_M __m, _N __n)

// Remarks: If either M or N is not an integer type,
// or if either is (a possibly cv-qualified) bool, the program is ill-formed.

#include <cuda/std/__numeric_>

int main(int, char**)
{
  cuda::std::lcm<volatile bool, int>(false, 4);

  return 0;
}
