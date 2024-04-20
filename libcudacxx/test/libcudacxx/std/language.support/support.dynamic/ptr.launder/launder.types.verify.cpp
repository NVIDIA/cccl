//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <new>

// template <class T> constexpr T* launder(T* p) noexcept;
// The program is ill-formed if T is a function type or cv void.

#include <cuda/std/__new_>
#include <cuda/std/cassert>

#include "test_macros.h"

__host__ __device__ void foo() {}

int main(int, char**)
{
  void* p = nullptr;
  (void) cuda::std::launder((void*) nullptr);
  (void) cuda::std::launder((const void*) nullptr);
  (void) cuda::std::launder((volatile void*) nullptr);
  (void) cuda::std::launder((const volatile void*) nullptr); // expected-error-re@new:* 4 {{static assertion
                                                             // failed{{.*}}can't launder cv-void}}
  // expected-error@new:* 0-4 {{void pointer argument to '__builtin_launder' is not allowed}}

  (void) cuda::std::launder(foo); // expected-error-re@new:* 1 {{static assertion failed{{.*}}can't launder functions}}
  // expected-error@new:* 0-1 {{function pointer argument to '__builtin_launder' is not allowed}}

  return 0;
}
