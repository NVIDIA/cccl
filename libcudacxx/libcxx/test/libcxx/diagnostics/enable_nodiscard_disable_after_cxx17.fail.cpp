// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// GCC 7 is the first version to introduce [[nodiscard]]
// UNSUPPORTED: gcc-5, gcc-6

// Test that _LIBCUDACXX_DISABLE_NODISCARD_EXT only disables _LIBCUDACXX_NODISCARD_EXT
// and not _LIBCUDACXX_NODISCARD_AFTER_CXX17.

// MODULES_DEFINES: _LIBCUDACXX_ENABLE_NODISCARD
// MODULES_DEFINES: _LIBCUDACXX_DISABLE_NODISCARD_AFTER_CXX17
#define _LIBCUDACXX_ENABLE_NODISCARD
#define _LIBCUDACXX_DISABLE_NODISCARD_AFTER_CXX17
#include <__config>


_LIBCUDACXX_NODISCARD_EXT int foo() { return 42; }
_LIBCUDACXX_NODISCARD_AFTER_CXX17 int bar() { return 42; }

int main(int, char**) {
  foo(); // expected-error-re {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  bar(); // OK.
  (void)foo(); // OK.

  return 0;
}
