//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <list>

// void clear() noexcept;

#include <list>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
    int a[] = {1, 2, 3};
    std::list<int> c(a, a+3);
    ASSERT_NOEXCEPT(c.clear());
    c.clear();
    assert(c.empty());
    }
    {
    int a[] = {1, 2, 3};
    std::list<int, min_allocator<int>> c(a, a+3);
    ASSERT_NOEXCEPT(c.clear());
    c.clear();
    assert(c.empty());
    }

  return 0;
}
