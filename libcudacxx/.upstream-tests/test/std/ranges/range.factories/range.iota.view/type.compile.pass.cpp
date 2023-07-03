//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

#include <cuda/std/ranges>

// Test that we SFINAE away iota_view<bool>.

template<class T> __host__ __device__ cuda::std::ranges::iota_view<T> f(int);
template<class T> __host__ __device__ void f(...) {}

__host__ __device__ void test() {
  f<bool>(42);
}

int main(int, char**) {
  return 0;
}

