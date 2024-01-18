//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <mutex>

// struct once_flag;

// constexpr once_flag() noexcept;

#include <mutex>
#include "test_macros.h"

int main(int, char**)
{
    {
    std::once_flag f;
    (void)f;
    }
    {
    constexpr std::once_flag f;
    (void)f;
    }

  return 0;
}
