//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// Verify TEST_WORKAROUND_C1XX_BROKEN_ZA_CTOR_CHECK.

#include <cuda/std/type_traits>

#include "test_macros.h"
#include "test_workarounds.h"

struct X
{
  __host__ __device__ X(int) {}

  X(X&&)            = default;
  X& operator=(X&&) = default;

private:
  X(const X&)            = default;
  X& operator=(const X&) = default;
};

__host__ __device__ void PushFront(X&&) {}

template <class T = int>
__host__ __device__ auto test(int) -> decltype(PushFront(cuda::std::declval<T>()), cuda::std::true_type{});
__host__ __device__ auto test(long) -> cuda::std::false_type;

int main(int, char**)
{
#if defined(TEST_WORKAROUND_C1XX_BROKEN_ZA_CTOR_CHECK)
  static_assert(!decltype(test(0))::value, "");
#else
  static_assert(decltype(test(0))::value, "");
#endif

  return 0;
}
