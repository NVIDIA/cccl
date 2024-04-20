//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <memory>

// struct allocator_arg_t { explicit allocator_arg_t() = default; };
// const allocator_arg_t allocator_arg = allocator_arg_t();

#include <cuda/std/__memory_>

#include "test_macros.h"

__host__ __device__ void test(cuda::std::allocator_arg_t) {}

int main(int, char**)
{
  test(cuda::std::allocator_arg);

  return 0;
}
