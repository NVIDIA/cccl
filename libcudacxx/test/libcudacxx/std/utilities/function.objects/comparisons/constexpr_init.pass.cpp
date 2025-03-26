//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: gcc-7 && !nvrtc
// GCC 5: Fails for C++11, passes for C++14.
// GCC 6: Fails for C++11, passes for C++14.
// GCC 7: Fails for C++11, fails for C++14.
// GCC 8: Fails for C++11, passes for C++14.

// XFAIL: msvc-19.0

// <cuda/std/functional>

// equal_to, not_equal_to, less, et al.

// Test that these types can be constructed w/o an initializer in a constexpr
// context. This is specifically testing gcc.gnu.org/PR83921

#include <cuda/std/functional>

#include "test_macros.h"

template <class T>
__host__ __device__ constexpr bool test_constexpr_context()
{
  [[maybe_unused]] cuda::std::equal_to<T> eq;
  [[maybe_unused]] cuda::std::not_equal_to<T> neq;
  [[maybe_unused]] cuda::std::less<T> l;
  [[maybe_unused]] cuda::std::less_equal<T> le;
  [[maybe_unused]] cuda::std::greater<T> g;
  [[maybe_unused]] cuda::std::greater_equal<T> ge;
  return true;
}

static_assert(test_constexpr_context<int>(), "");
static_assert(test_constexpr_context<void>(), "");

int main(int, char**)
{
  return 0;
}
