//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// test set_new_handler

#include <cuda/std/__new>
#include <cuda/std/cassert>

#include "test_macros.h"

__host__ __device__ void f1() {}
__host__ __device__ void f2() {}

int main(int, char**)
{
    assert(cuda::std::set_new_handler(f1) == 0);
    assert(cuda::std::set_new_handler(f2) == f1);

  return 0;
}
