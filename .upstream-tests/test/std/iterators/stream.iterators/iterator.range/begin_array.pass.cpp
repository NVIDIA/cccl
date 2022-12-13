//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// template <class T, size_t N> T* begin(T (&array)[N]);

#include <cuda/std/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    int ia[] = {1, 2, 3};
    int* i = cuda::std::begin(ia);
    assert(*i == 1);
    *i = 2;
    assert(ia[0] == 2);

  return 0;
}
