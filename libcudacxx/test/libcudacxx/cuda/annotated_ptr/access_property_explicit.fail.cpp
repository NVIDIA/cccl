//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-70
// UNSUPPORTED: !nvcc

// UNSUPPORTED: c++98, c++03

#include "utils.h"

__host__ __device__ __noinline__ void test_access_property_fail()
{
  cuda::access_property o = cuda::access_property::normal{};
  // Test implicit conversion fails
  std::uint64_t x;
  x = o;
  unused(o);
}

int main(int argc, char** argv)
{
  test_access_property_fail();
  return 0;
}
