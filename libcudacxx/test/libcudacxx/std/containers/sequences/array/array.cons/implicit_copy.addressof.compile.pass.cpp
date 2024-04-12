//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// implicitly generated array assignment operators

// Validate whether the container can be copy-assigned with an ADL-hijacking operator&

#include <cuda/std/array>

#include "operator_hijacker.h"
#include "test_macros.h"

__host__ __device__ void test()
{
  cuda::std::array<operator_hijacker, 1> ao{};
  cuda::std::array<operator_hijacker, 1> a;
  a = ao;
  unused(a);
}

int main(int, char**)
{
  return 0;
}
