//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: true

// test get_terminate

#include <cuda/std/__exception/terminate.h>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

__host__ __device__ void f1() {}
__host__ __device__ void f2() {}

int main(int, char**)
{
  cuda::std::set_terminate(f1);
  assert(cuda::std::get_terminate() == f1);
  cuda::std::set_terminate(f2);
  assert(cuda::std::get_terminate() == f2);

  return 0;
}
