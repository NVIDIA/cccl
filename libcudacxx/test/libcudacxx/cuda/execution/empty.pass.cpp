//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__cccl/attributes.h>

struct Empty
{};

template <class T>
struct Wrapper
{
  _CCCL_NO_UNIQUE_ADDRESS T value;
};

__host__ __device__ void test()
{
  [[maybe_unused]] Wrapper<Empty> w{};
}

int main(int, char**)
{
  test();
  return 0;
}
