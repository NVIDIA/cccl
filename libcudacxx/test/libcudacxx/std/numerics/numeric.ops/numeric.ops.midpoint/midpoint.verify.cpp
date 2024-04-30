//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// <numeric>

// template <class _Tp>
// _Tp midpoint(_Tp __a, _Tp __b) noexcept

// An overload exists for each of char and all arithmetic types except bool.

#include <cuda/std/numeric>

#include "test_macros.h"

__host__ __device__ int func1()
{
  return 1;
}
__host__ __device__ int func2()
{
  return 2;
}

struct Incomplete;
Incomplete* ip = nullptr;
void* vp       = nullptr;

int main(int, char**)
{
  (void) cuda::std::midpoint(false, true); // expected-error {{no matching function for call to 'midpoint'}}

  //  A couple of odd pointer types that should fail
  (void) cuda::std::midpoint(nullptr, nullptr); // expected-error {{no matching function for call to 'midpoint'}}
  (void) cuda::std::midpoint(func1, func2); // expected-error {{no matching function for call to 'midpoint'}}
  (void) cuda::std::midpoint(ip, ip); // expected-error {{no matching function for call to 'midpoint'}}
  (void) cuda::std::midpoint(vp, vp); // expected-error {{no matching function for call to 'midpoint'}}

  return 0;
}
