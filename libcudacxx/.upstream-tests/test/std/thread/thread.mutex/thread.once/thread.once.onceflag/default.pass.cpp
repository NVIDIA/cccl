//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// UNSUPPORTED: pre-sm-70

// <mutex>

// struct once_flag;

// constexpr once_flag() noexcept;

#include<cuda/std/mutex>
#include "test_macros.h"

int main(int, char**)
{
    {
    cuda::std::once_flag f;
    unused(f);
    }
#if TEST_STD_VER >= 11
    {
    constexpr cuda::std::once_flag f;
    unused(f);
    }
#endif

  return 0;
}
