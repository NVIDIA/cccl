//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// test new_handler

#include <cuda/std/__new>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"

__host__ __device__ void f() {}

int main(int, char**)
{
    static_assert((cuda::std::is_same<cuda::std::new_handler, void(*)()>::value), "");
    cuda::std::new_handler p = f;
    assert(p == &f);

  return 0;
}
