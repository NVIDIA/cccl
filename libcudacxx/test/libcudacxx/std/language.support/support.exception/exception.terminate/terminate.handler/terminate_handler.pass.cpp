//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: true

// test terminate_handler

#include <cuda/std/__exception/terminate.h>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

__host__ __device__ void f() {}

int main(int, char**)
{
  static_assert((cuda::std::is_same<cuda::std::terminate_handler, void (*)()>::value), "");
  cuda::std::terminate_handler p = f;
  assert(p == &f);

  return 0;
}
