//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <new>

// template <class T> constexpr T* launder(T* p) noexcept;

#include <cuda/std/__new_>

__host__ __device__ void f()
{
  int* p = nullptr;
  cuda::std::launder(p); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
