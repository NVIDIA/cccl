//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: nvrtc

// <memory>

// allocator:
// T* allocate(size_t n);

#include <cuda/std/__memory_>

__host__ __device__ void f()
{
  cuda::std::allocator<int> a;
  a.allocate(3); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}

int main(int, char**)
{
  return 0;
}
