//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// unary_negate
//  deprecated in C++17

// UNSUPPORTED: clang-4.0
// UNSUPPORTED: c++98, c++03, c++11, c++14
// REQUIRES: verify-support

#include <cuda/std/functional>

#include "test_macros.h"

struct Predicate
{
  typedef int argument_type;
  bool operator()(argument_type) const
  {
    return true;
  }
};

int main(int, char**)
{
  cuda::std::unary_negate<Predicate> f((Predicate())); // expected-error{{'unary_negate<Predicate>' is deprecated}}
  (void) f;

  return 0;
}
