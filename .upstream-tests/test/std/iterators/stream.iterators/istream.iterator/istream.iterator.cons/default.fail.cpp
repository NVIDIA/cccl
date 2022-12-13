//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// UNSUPPORTED: nvrtc

// class istream_iterator

// constexpr istream_iterator();

#include <cuda/std/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"

struct S { S(); }; // not constexpr

int main(int, char**)
{
#if TEST_STD_VER >= 11
    {
    constexpr cuda::std::istream_iterator<S> it;
    }
#else
#error "C++11 only test"
#endif

  return 0;
}
