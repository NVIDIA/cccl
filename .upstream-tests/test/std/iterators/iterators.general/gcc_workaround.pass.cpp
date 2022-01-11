//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Tests workaround for  https://gcc.gnu.org/bugzilla/show_bug.cgi?id=64816.

#include <cuda/std/string>
#include "test_macros.h"

void f(const cuda::std::string &s) { TEST_IGNORE_NODISCARD s.begin(); }

#include <cuda/std/vector>

void AppendTo(const cuda::std::vector<char> &v) { TEST_IGNORE_NODISCARD v.begin(); }

int main(int, char**) {
  return 0;
}
