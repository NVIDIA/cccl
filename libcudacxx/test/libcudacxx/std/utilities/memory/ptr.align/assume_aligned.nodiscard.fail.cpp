//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// #include <memory>

// template<size_t N, class T>
// [[nodiscard]] constexpr T* assume_aligned(T* ptr);

// UNSUPPORTED: nvrtc
// nvrtc currently compiles the test with a warning

#include <cuda/std/memory>

__host__ __device__ void f()
{
  int* p = nullptr;
  cuda::std::assume_aligned<4>(p); // expected-warning {{ignoring return value of function declared with 'nodiscard'
                                   // attribute}}
}

int main(int, char**)
{
  return 0;
}
