
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// <concepts> feature macros

/*  Constant                                    Value
    __cpp_lib_concepts                          201806L

*/

#include <concepts>
#include <cassert>
#include "test_macros.h"

int main(int, char**)
{
//  ensure that the macros that are supposed to be defined in <concepts> are defined.

#if TEST_STD_VER < 14

# ifdef __cpp_lib_concepts
#   error "__cpp_lib_concepts should not be defined before c++14"
# endif

#else
#if !defined(__cpp_lib_concepts)
# error "__cpp_lib_concepts is not defined"
#elif __cpp_lib_concepts < 202002L
# error "__cpp_lib_concepts has an invalid value"
#endif
#endif // TEST_STD_VER < 14

  return 0;
}
