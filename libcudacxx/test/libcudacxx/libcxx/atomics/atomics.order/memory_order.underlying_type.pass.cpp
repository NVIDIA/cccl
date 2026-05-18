//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// This test ensures that cuda::std::memory_order has the same size under all
// standard versions to make sure we're not breaking the ABI. This is
// relevant because cuda::std::memory_order is a scoped enumeration in C++20,
// but an unscoped enumeration pre-C++20.
//
// See PR40977 for details.

#include <cuda/std/atomic>
#include <cuda/std/type_traits>

#include "test_macros.h"

enum cpp17_memory_order
{
  cpp17_memory_order_relaxed,
  cpp17_memory_order_consume,
  cpp17_memory_order_acquire,
  cpp17_memory_order_release,
  cpp17_memory_order_acq_rel,
  cpp17_memory_order_seq_cst
};

static_assert((cuda::std::is_same<cuda::std::underlying_type<cpp17_memory_order>::type,
                                  cuda::std::underlying_type<cuda::std::memory_order>::type>::value),
              "cuda::std::memory_order should have the same underlying type as a corresponding "
              "unscoped enumeration would. Otherwise, our ABI changes from C++17 to C++20.");

int main(int, char**)
{
  return 0;
}
