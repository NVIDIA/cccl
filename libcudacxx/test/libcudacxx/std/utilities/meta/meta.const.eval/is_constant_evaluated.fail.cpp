//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <cuda/std/type_traits>

#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
#ifndef _CCCL_BUILTIN_IS_CONSTANT_EVALUATED
  // expected-error@+1 {{no member named 'is_constant_evaluated' in namespace 'std'}}
  bool b = cuda::std::is_constant_evaluated();
#else
  // expected-error-re@+1 {{{{(static_assert|static assertion)}} failed}}
  static_assert(!cuda::std::is_constant_evaluated(), "");
#endif
  return 0;
}
