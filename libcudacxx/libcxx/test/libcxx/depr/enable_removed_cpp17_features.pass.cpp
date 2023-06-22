//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that defining _LIBCUDACXX_ENABLE_CXX17_REMOVED_FEATURES correctly defines
// _LIBCUDACXX_ENABLE_CXX17_REMOVED_FOO for each individual component macro.

// MODULES_DEFINES: _LIBCUDACXX_ENABLE_CXX17_REMOVED_FEATURES
#define _LIBCUDACXX_ENABLE_CXX17_REMOVED_FEATURES
#include <__config>

#include "test_macros.h"

#ifndef _LIBCUDACXX_ENABLE_CXX17_REMOVED_UNEXPECTED_FUNCTIONS
#error _LIBCUDACXX_ENABLE_CXX17_REMOVED_UNEXPECTED_FUNCTIONS must be defined
#endif

#ifndef _LIBCUDACXX_ENABLE_CXX17_REMOVED_AUTO_PTR
#error _LIBCUDACXX_ENABLE_CXX17_REMOVED_AUTO_PTR must be defined
#endif

int main(int, char**) {

  return 0;
}
