//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Due to C++17 inline variables ASAN flags this test as containing an ODR
// violation because Clock::is_steady is defined in both the dylib and this TU.
// UNSUPPORTED: asan

// <cuda/std/chrono>

// system_clock

// check clock invariants

#include <cuda/std/chrono>

template <class T>
__host__ __device__ void test(const T&)
{}

int main(int, char**)
{
  typedef cuda::std::chrono::system_clock C;
  static_assert((cuda::std::is_same<C::rep, C::duration::rep>::value), "");
  static_assert((cuda::std::is_same<C::period, C::duration::period>::value), "");
  static_assert((cuda::std::is_same<C::duration, C::time_point::duration>::value), "");
  static_assert((cuda::std::is_same<C::time_point::clock, C>::value), "");
  static_assert((C::is_steady || !C::is_steady), "");
  test(+cuda::std::chrono::system_clock::is_steady);

  return 0;
}
